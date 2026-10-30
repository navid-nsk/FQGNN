# -*- coding: utf-8 -*-
# Portions of this file are adapted from MultitaskAIS (https://github.com/dnguyengithub/MultitaskAIS):
#   create_AIS_dataset_native   <- data/datasets.py, create_AIS_dataset     (MIT License)
#   _build_model, the training graph of run_train
#                               <- runners.py, create_dataset_and_model / run_train
#   create_eval_graph_windowed  <- runners.py, create_eval_graph            (Apache License 2.0)
# These portions remain under their original licenses (notices reproduced in
# task_alpha/baselines/THIRD_PARTY.md); every change to them is listed in the docstring below.
"""Stage 2 (mtais_tf1 env: Python 3.7, TensorFlow 1.15 CPU, dm-sonnet 1.36): train the vendor VRNN
and run trajectory reconstruction on one exported cell + seed.

Run ONLY via:  conda run -n mtais_tf1 python -u mtais_tf1_run.py ...   (run_mtais_gaps.py does this)
(py3.7 syntax throughout; no torch; vendor code imported from MultitaskAIS-master, NEVER modified.)

VENDOR FIDELITY
---------------
Training is runners.run_train re-expressed with explicit step accounting: same graph construction
(create_vrnn(latent, ConditionalBernoulliDistribution, generative_bias_init from mean.pkl,
raw_sigma_bias=0.5)), same objective (bounds.elbo, num_samples=1, loss = -mean(ll_per_seq/len)),
same optimizer (AdamOptimizer(3e-4), compute_gradients/apply_gradients), same session type
(MonitoredTrainingSession, checkpoints every 120 s + at close), tf.set_random_seed(seed).
README hyperparameters: latent_size=100, batch_size=32, lr=3e-4; reconstruction batch_size=1,
num_samples=16, bound=elbo (the vendor's own traj_reconstruction invocation).

The data pipeline is the vendor's datasets.create_AIS_dataset cloned with exactly TWO changes:
  (a) the voyage dict is passed in-memory instead of read from dataset_path (no semantic change);
  (b) the [::2] cadence-halving in aistrack_generator is REMOVED.  That slice exists because the
      vendor's preprocessed data is 5-min resampled while the paper operates at 10 min; our exports
      are already at the benchmark's native cadence, so keeping it would silently halve every
      sequence and misalign the gap window.  Data-cadence adaptation, not a model change.
Everything else (four-hot py_func, padded_batch, shuffle(num_examples), the pad-shift input
construction of process_AIS_batch, prefetch(50)) is byte-equivalent to the vendor code.

RECONSTRUCTION = the vendor traj_reconstruction mechanism (runners.create_eval_graph copied below),
generalized in ONLY three ways:
  (i)  the missing-data window is [gap_start, gap_end) from test_index.pkl instead of the hardcoded
       [max_seq_len-18, max_seq_len-6);
  (ii) the held length is the cell's own gap length (n_held), not 12 steps;
  (iii) numeric output instead of plots: 16 samples, argmax lat/lon bin per sample per held step,
       written to predictions.npz.  The a-priori aggregation rule (fixed BEFORE any scoring) is the
       MEAN over the 16 samples per step -- never best-of-sample.
With batch_size=1 (the vendor's own reconstruction batch size) the scalar tf.cond window is exactly
the vendor mechanism.  Inside the window BOTH the input and the target slots are replaced by
(model sample, zeros) -- vendor behaviour: the zero target trips no_target and the model samples
from its PRIOR.  bound=elbo means no resampling, so held targets never influence the samples.

ONE-SIDED PROTOCOL (stated explicitly): this model is a forward LSTM sampling from its prior
inside the gap.  It is open-loop autoregressive forecasting seeded only by the pre-gap history.  We
do NOT feed post-gap anchors, do NOT reverse or symmetrize -- MultitaskAIS is reported as a
separate one-sided reference, not as a two-sided peer.

TRAINING BUDGET: min(20 epochs, 30000 steps) at batch 32 (with our capped exports 20 epochs is
always the binding constraint; the vendor default is max_steps=80000 with no other stop).  No
early stopping and no validation-based selection -- the vendor has neither.

Held steps are placed by COUNT (the four-hot encoding has no time channel) -- see meta.json.
"""
from __future__ import absolute_import, division, print_function

import argparse
import csv
import json
import math
import os
import pickle
import shutil
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
VENDOR = os.path.abspath(os.path.join(HERE, "..", "MultitaskAIS-master"))
if VENDOR not in sys.path:
    sys.path.insert(0, VENDOR)

import tensorflow as tf  # noqa: E402  (1.15)

import bounds  # noqa: E402  vendor
import nested_utils as nested  # noqa: E402  vendor
import distribution_utils as dists  # noqa: E402  vendor
from models import vrnn  # noqa: E402  vendor

LAT, LON, SOG, COG, HEADING, ROT, NAV_STT, TIMESTAMP, MMSI = list(range(9))
DEFAULT_PARALLELISM = 12               # vendor datasets.py


# ------------------------------------------------------------------------------------------------ #
#  numpy-2 pickle compatibility (exports are written by numpy 2.x; this env runs numpy 1.16)         #
# ------------------------------------------------------------------------------------------------ #
class _NpCompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core", 1)
        return pickle.Unpickler.find_class(self, module, name)


def _np_compat_load(path):
    with open(path, "rb") as f:
        return _NpCompatUnpickler(f).load()


# ------------------------------------------------------------------------------------------------ #
#  four-hot encoding (vendor create_dense_vect, vectorized -- identical bins)                        #
# ------------------------------------------------------------------------------------------------ #
def four_hot(rows, lat_bins, lon_bins, sog_bins, cog_bins):
    v = np.clip(np.asarray(rows, np.float64), 0.0, 0.99999)
    T = v.shape[0]
    D = lat_bins + lon_bins + sog_bins + cog_bins
    out = np.zeros((T, D), dtype=np.float32)
    r = np.arange(T)
    out[r, (v[:, 0] * lat_bins).astype(np.int64)] = 1.0
    out[r, (v[:, 1] * lon_bins).astype(np.int64) + lat_bins] = 1.0
    out[r, (v[:, 2] * sog_bins).astype(np.int64) + lat_bins + lon_bins] = 1.0
    out[r, (v[:, 3] * cog_bins).astype(np.int64) + lat_bins + lon_bins + sog_bins] = 1.0
    return out


# ------------------------------------------------------------------------------------------------ #
#  vendor datasets.create_AIS_dataset, cloned (dict input; [::2] removed -- see module docstring)    #
# ------------------------------------------------------------------------------------------------ #
def create_AIS_dataset_native(raw_data, mean, batch_size, data_dim,
                              lat_bins, lon_bins, sog_bins, cog_bins,
                              num_parallel_calls=DEFAULT_PARALLELISM, shuffle=True, repeat=True):
    total_bins = lat_bins + lon_bins + sog_bins + cog_bins

    def sparse_AIS_to_dense(msgs_, num_timesteps, mmsis, time_start, time_end):
        dense_msgs = four_hot(msgs_, lat_bins, lon_bins, sog_bins, cog_bins).astype(np.float64)
        return dense_msgs, num_timesteps, mmsis, time_start, time_end

    num_examples = len(raw_data)

    def aistrack_generator():
        for k in list(raw_data.keys()):
            tmp = raw_data[k][:, [LAT, LON, SOG, COG]]          # native cadence (vendor: [::2])
            tmp[tmp == 1] = 0.99999
            yield tmp, len(tmp), raw_data[k][0, MMSI], raw_data[k][0, TIMESTAMP], \
                raw_data[k][-1, TIMESTAMP]

    dataset = tf.data.Dataset.from_generator(
        aistrack_generator,
        output_types=(tf.float64, tf.int64, tf.int64, tf.float32, tf.float32))

    if repeat:
        dataset = dataset.repeat()
    if shuffle:
        dataset = dataset.shuffle(num_examples)

    dataset = dataset.map(
        lambda msg_, num_timesteps, mmsis, time_start, time_end: tuple(tf.py_func(
            sparse_AIS_to_dense,
            [msg_, num_timesteps, mmsis, time_start, time_end],
            [tf.float64, tf.int64, tf.int64, tf.float32, tf.float32])),
        num_parallel_calls=num_parallel_calls)

    dataset = dataset.padded_batch(batch_size, padded_shapes=([None, total_bins], [], [], [], []))

    def process_AIS_batch(data, lengths, mmsis, time_start, time_end):
        """Create mean-centered and time-major next-step prediction Tensors (vendor verbatim)."""
        data = tf.to_float(tf.transpose(data, perm=[1, 0, 2]))
        lengths = tf.to_int32(lengths)
        mmsis = tf.to_int32(mmsis)
        targets = data
        # Mean center the inputs.
        inputs = data - tf.constant(mean, dtype=tf.float32, shape=[1, 1, mean.shape[0]])
        # Shift the inputs one step forward in time. Also remove the last
        # timestep so that targets and inputs are the same length.
        inputs = tf.pad(data, [[1, 0], [0, 0], [0, 0]], mode="CONSTANT")[:-1]
        # Mask out unused timesteps.
        inputs *= tf.expand_dims(tf.transpose(tf.sequence_mask(lengths, dtype=inputs.dtype)), 2)
        return inputs, targets, lengths, mmsis, time_start, time_end

    dataset = dataset.map(process_AIS_batch, num_parallel_calls=num_parallel_calls)
    dataset = dataset.prefetch(50)
    itr = dataset.make_one_shot_iterator()
    inputs, targets, lengths, mmsis, time_starts, time_ends = itr.get_next()
    return inputs, targets, mmsis, time_starts, time_ends, lengths, tf.constant(mean, dtype=tf.float32)


def _build_model(data_dim, latent_size, mean):
    """runners.create_dataset_and_model model construction, verbatim."""
    generative_bias_init = -tf.log(1. / tf.clip_by_value(tf.constant(mean, dtype=tf.float32),
                                                         0.0001, 0.9999) - 1)
    return vrnn.create_vrnn(data_dim, latent_size, vrnn.ConditionalBernoulliDistribution,
                            generative_bias_init=generative_bias_init, raw_sigma_bias=0.5)


# ------------------------------------------------------------------------------------------------ #
#  TRAIN: runners.run_train with explicit step accounting + capped budget                            #
# ------------------------------------------------------------------------------------------------ #
def run_train(export_dir, out_dir, seed, epochs, max_steps, batch_size, latent_size, lr):
    raw_data = _np_compat_load(os.path.join(export_dir, "train.pkl"))
    mean = np.asarray(_np_compat_load(os.path.join(export_dir, "mean.pkl")), np.float64)
    with open(os.path.join(export_dir, "meta.json")) as f:
        meta = json.load(f)
    roi = meta["roi"]
    data_dim = int(roi["data_dim"])
    assert mean.shape[0] == data_dim

    n_voy = len(raw_data)
    steps_per_epoch = max(1, int(math.ceil(n_voy / float(batch_size))))
    total_steps = min(epochs * steps_per_epoch, max_steps)
    chkpt_dir = os.path.join(out_dir, "chkpt")
    print("[mtais-tf1] train: %d voyages, %d steps/epoch, budget min(%d ep, %d steps) = %d steps, "
          "data_dim %d, latent %d, batch %d, lr %g, seed %d"
          % (n_voy, steps_per_epoch, epochs, max_steps, total_steps, data_dim,
             latent_size, batch_size, lr, seed), flush=True)

    if os.path.isdir(chkpt_dir):
        shutil.rmtree(chkpt_dir)          # fresh, seed-clean training (no partial resume)
    os.makedirs(chkpt_dir)

    t0 = time.time()
    ep_rows = []
    with tf.Graph().as_default():
        tf.set_random_seed(seed)
        np.random.seed(seed)
        inputs, targets, mmsis, ts, te_, lengths, _ = create_AIS_dataset_native(
            raw_data, mean, batch_size, data_dim,
            int(roi["lat_bins"]), int(roi["lon_bins"]), int(roi["sog_bins"]), int(roi["cog_bins"]),
            shuffle=True, repeat=True)
        model = _build_model(data_dim, latent_size, mean)
        global_step = tf.train.get_or_create_global_step()
        # vendor run_train.create_loss, bound == "elbo", normalize_by_seq_len == True:
        ll_per_seq, _, _, _ = bounds.elbo(model, (inputs, targets), lengths, num_samples=1)
        ll_per_t = tf.reduce_mean(ll_per_seq / tf.to_float(lengths))
        loss = -ll_per_t
        opt = tf.train.AdamOptimizer(lr)
        grads = opt.compute_gradients(loss, var_list=tf.trainable_variables())
        train_op = opt.apply_gradients(grads, global_step=global_step)

        step, acc, nacc = 0, 0.0, 0
        with tf.train.MonitoredTrainingSession(checkpoint_dir=chkpt_dir,
                                               save_checkpoint_secs=120,
                                               save_summaries_steps=None,
                                               save_summaries_secs=None,
                                               log_step_count_steps=10 ** 9) as sess:
            while step < total_steps:
                _, b = sess.run([train_op, ll_per_t])
                step += 1
                acc += float(b)
                nacc += 1
                if step % steps_per_epoch == 0 or step == total_steps:
                    ep = len(ep_rows)
                    ep_rows.append({"epoch": ep, "steps": step,
                                    "train_ll_per_t": round(acc / max(nacc, 1), 5),
                                    "seconds": round(time.time() - t0, 1)})
                    print("[mtais-tf1] seed %d epoch %02d (step %d/%d)  elbo/t %.4f  (%.0fs)"
                          % (seed, ep, step, total_steps, acc / max(nacc, 1), time.time() - t0),
                          flush=True)
                    acc, nacc = 0.0, 0

    with open(os.path.join(out_dir, "loss_curve.csv"), "w") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "steps", "train_ll_per_t", "seconds"],
                           lineterminator="\n")
        w.writeheader()
        w.writerows(ep_rows)
    train_meta = {"completed": True, "seed": seed, "n_train_voyages": n_voy,
                  "steps_per_epoch": steps_per_epoch, "total_steps": total_steps,
                  "epochs_run": len(ep_rows), "budget": "min(%d epochs, %d steps)" % (epochs, max_steps),
                  "batch_size": batch_size, "latent_size": latent_size, "learning_rate": lr,
                  "num_samples_train": 1, "bound": "elbo",
                  "final_train_ll_per_t": ep_rows[-1]["train_ll_per_t"] if ep_rows else None,
                  "train_seconds": round(time.time() - t0, 1),
                  "tf_version": tf.__version__}
    with open(os.path.join(out_dir, "train_meta.json"), "w") as f:
        json.dump(train_meta, f, indent=1)
    print("[mtais-tf1] training done: %d steps in %.0fs" % (total_steps, time.time() - t0), flush=True)
    return train_meta


# ------------------------------------------------------------------------------------------------ #
#  EVAL GRAPH: runners.create_eval_graph copied VERBATIM except the gap window                       #
#  (vendor: t in [max_seq_len-18, max_seq_len-6) hardcoded; here: [gap_start, gap_end) tensors).     #
# ------------------------------------------------------------------------------------------------ #
def create_eval_graph_windowed(inputs, targets, lengths, model, config, gap_start, gap_end):
    parallel_iterations = 30
    swap_memory = True
    batch_size = tf.shape(lengths)[0]
    num_samples = config.num_samples
    max_seq_len = tf.reduce_max(lengths)
    init_states = model.zero_state(batch_size * num_samples, tf.float32)

    seq_mask = tf.transpose(
        tf.sequence_mask(lengths, maxlen=max_seq_len, dtype=tf.float32),
        perm=[1, 0])
    if num_samples > 1:
        inputs_tmp, seq_mask = nested.tile_tensors([(inputs, targets), seq_mask], [1, num_samples])
        inputs_ta, mask_ta = nested.tas_for_tensors([inputs_tmp, seq_mask], max_seq_len)
    else:
        inputs_ta, mask_ta = nested.tas_for_tensors([(inputs, targets), seq_mask], max_seq_len)

    t0 = tf.constant(0, tf.int32)
    init_states = model.zero_state(batch_size * num_samples, tf.float32)

    ta_names = ['log_weights_t', 'sampleds', 'trues', 'rnn_states', 'rnn_latents', 'rnn_outs']
    tas = [tf.TensorArray(tf.float32, max_seq_len, name='%s_ta' % n) for n in ta_names]

    log_weights_acc = tf.zeros([num_samples, batch_size], dtype=tf.float32)
    log_p_hat_acc = tf.zeros([batch_size], dtype=tf.float32)
    kl_acc = tf.zeros([num_samples * batch_size], dtype=tf.float32)
    if config.bound == "elbo":
        accs = (log_weights_acc, kl_acc)
    elif config.bound == "fivo":
        accs = (log_weights_acc, log_p_hat_acc, kl_acc)

    target_sampled0 = tf.zeros(shape=[batch_size * num_samples, config.data_dim], dtype=tf.float32)
    target_true0 = tf.zeros(shape=[batch_size * num_samples, config.data_dim], dtype=tf.float32)
    init_targets = (target_sampled0, target_true0)

    def while_predicate(t, *unused_args):
        return t < max_seq_len

    resampling_criterion = bounds.ess_criterion

    def while_step(t, rnn_state, tas, accs, while_samples):
        """Implements one timestep of IWAE computation."""
        if config.bound == "elbo":
            log_weights_acc, kl_acc = accs
        elif config.bound == "fivo":
            log_weights_acc, log_p_hat_acc, kl_acc = accs
        cur_inputs, cur_mask = nested.read_tas([inputs_ta, mask_ta], t)

        if config.missing_data:
            # GENERALIZATION (i)+(ii): vendor line is
            #   tf.cond(tf.logical_and(t < max_seq_len - 6, t >= max_seq_len - 18), ...)
            cur_inputs = tf.cond(tf.logical_and(t < gap_end, t >= gap_start),
                                 lambda: while_samples,
                                 lambda: cur_inputs)

        log_q_z, log_p_z, log_p_x_given_z, kl, new_state, new_rnn_out, dists_return \
            = model(cur_inputs, rnn_state, cur_mask, return_value="probs")

        new_sample0 = dists.sample_from_probs(dists_return,
                                              config.onehot_lat_bins,
                                              config.onehot_lon_bins,
                                              config.onehot_sog_bins,
                                              config.onehot_cog_bins)
        new_sample0 = tf.cast(new_sample0, tf.float32)
        new_sample_ = (new_sample0, tf.zeros_like(new_sample0, dtype=tf.float32))

        kl_acc += kl * cur_mask
        log_alpha = (log_p_x_given_z + log_p_z - log_q_z) * cur_mask
        log_alpha = tf.reshape(log_alpha, [config.num_samples, batch_size])
        log_weights_acc += log_alpha
        ess_num = 2 * tf.reduce_logsumexp(log_weights_acc, axis=0)
        ess_denom = tf.reduce_logsumexp(2 * log_weights_acc, axis=0)
        log_ess = ess_num - ess_denom
        if config.bound == "fivo":
            resampling_dist = tf.contrib.distributions.Categorical(
                logits=tf.transpose(log_weights_acc, perm=[1, 0]))
            ancestor_inds = tf.stop_gradient(
                resampling_dist.sample(sample_shape=num_samples, seed=config.random_seed))
            offset = tf.expand_dims(tf.range(batch_size), 0)
            ancestor_inds = tf.reshape(ancestor_inds * batch_size + offset, [-1])
            noresample_inds = tf.range(num_samples * batch_size)
            should_resample = resampling_criterion(num_samples, log_ess, t)
            should_resample = tf.logical_and(should_resample, cur_mask[:batch_size] > 0.)
            float_should_resample = tf.to_float(should_resample)
            ancestor_inds = tf.where(tf.tile(should_resample, [num_samples]),
                                     ancestor_inds, noresample_inds)
            new_state = nested.gather_tensors(new_state, ancestor_inds)
            new_sample_ = nested.gather_tensors(new_sample_, ancestor_inds)

        ta_updates = [log_alpha, new_sample_[0], new_sample_[1],
                      new_state[0], new_state[1], new_rnn_out]
        new_tas = [ta.write(t, x) for ta, x in zip(tas, ta_updates)]

        if config.bound == "fivo":
            log_p_hat_update = tf.reduce_logsumexp(
                log_weights_acc, axis=0) - tf.log(tf.to_float(num_samples))
            log_p_hat_acc += log_p_hat_update * float_should_resample
            log_weights_acc *= (1. - tf.tile(float_should_resample[tf.newaxis, :],
                                             [num_samples, 1]))
            new_accs = (log_weights_acc, log_p_hat_acc, kl_acc)
        elif config.bound == "elbo":
            new_accs = (log_weights_acc, kl_acc)
        return t + 1, new_state, new_tas, new_accs, new_sample_

    _, _, tas, accs, new_sample = tf.while_loop(while_predicate,
                                                while_step,
                                                loop_vars=(t0, init_states, tas, accs, init_targets),
                                                parallel_iterations=parallel_iterations,
                                                swap_memory=swap_memory)

    log_weights, track_sample, track_true, \
        rnn_state_tf, rnn_latent_tf, rnn_out_tf = [x.stack() for x in tas]

    if config.bound == "fivo":
        final_log_weights, log_p_hat, kl = accs
        log_p_hat += (tf.reduce_logsumexp(final_log_weights, axis=0) -
                      tf.log(tf.to_float(num_samples)))
        kl = tf.reduce_mean(tf.reshape(kl, [num_samples, batch_size]), axis=0)
    elif config.bound == "elbo":
        final_log_weights, kl = accs
        log_p_hat = (tf.reduce_logsumexp(final_log_weights, axis=0) -
                     tf.log(tf.to_float(num_samples)))
        kl = tf.reduce_mean(tf.reshape(kl, [num_samples, batch_size]), axis=0)

    ll_per_seq = log_p_hat
    ll_per_t = ll_per_seq / tf.to_float(lengths)
    return track_sample, track_true, log_weights, ll_per_t, \
        final_log_weights / tf.to_float(lengths), rnn_state_tf, rnn_latent_tf, rnn_out_tf


class _NS(object):
    def __init__(self, **kw):
        self.__dict__.update(kw)


# ------------------------------------------------------------------------------------------------ #
#  RECONSTRUCT: per test gap, feed pre-gap pings + gap window, collect 16 sampled four-hots per      #
#  held step, store argmax lat/lon bins.  Aggregation (mean over samples) happens in stage 3.        #
# ------------------------------------------------------------------------------------------------ #
def run_reconstruct(export_dir, out_dir, seed, num_samples, latent_size, test_cap):
    entries = _np_compat_load(os.path.join(export_dir, "test_index.pkl"))
    mean = np.asarray(_np_compat_load(os.path.join(export_dir, "mean.pkl")), np.float64)
    with open(os.path.join(export_dir, "meta.json")) as f:
        meta = json.load(f)
    roi = meta["roi"]
    lat_bins, lon_bins = int(roi["lat_bins"]), int(roi["lon_bins"])
    sog_bins, cog_bins = int(roi["sog_bins"]), int(roi["cog_bins"])
    data_dim = int(roi["data_dim"])
    chkpt_dir = os.path.join(out_dir, "chkpt")

    if test_cap:
        entries = entries[:test_cap]
        print("[mtais-tf1] SMOKE: test gaps capped to %d" % test_cap, flush=True)

    cfg = _NS(bound="elbo", missing_data=True, num_samples=num_samples, data_dim=data_dim,
              onehot_lat_bins=lat_bins, onehot_lon_bins=lon_bins,
              onehot_sog_bins=sog_bins, onehot_cog_bins=cog_bins, random_seed=seed)

    t0 = time.time()
    preds = {}
    n_done, n_skip = 0, 0
    with tf.Graph().as_default():
        tf.set_random_seed(seed)
        inputs_ph = tf.placeholder(tf.float32, [None, 1, data_dim], name="inputs")
        targets_ph = tf.placeholder(tf.float32, [None, 1, data_dim], name="targets")
        lengths_ph = tf.placeholder(tf.int32, [1], name="lengths")
        gs_ph = tf.placeholder(tf.int32, [], name="gap_start")
        ge_ph = tf.placeholder(tf.int32, [], name="gap_end")
        model = _build_model(data_dim, latent_size, mean)
        track_sample = create_eval_graph_windowed(inputs_ph, targets_ph, lengths_ph,
                                                  model, cfg, gs_ph, ge_ph)[0]
        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(chkpt_dir)
        if ckpt is None:
            raise RuntimeError("no checkpoint in %s -- train first" % chkpt_dir)
        with tf.Session() as sess:
            saver.restore(sess, ckpt)
            for e in entries:
                if e["skip"] or e["n_pre"] < 1 or e["n_held"] < 1:
                    n_skip += 1
                    continue
                pre = np.asarray(e["pre"], np.float64)
                n_pre, n_held = int(e["n_pre"]), int(e["n_held"])
                T = n_pre + n_held
                rows = np.concatenate([pre[:, [LAT, LON, SOG, COG]],
                                       np.repeat(pre[-1:, [LAT, LON, SOG, COG]], n_held, axis=0)],
                                      axis=0)
                # held rows are placeholder copies of the LAST OBSERVED ping.  They are never read:
                # inside [gap_start, gap_end) both input and target slots are replaced by
                # (sample, zeros), and the sequence ends at the gap end.  (Asserted leak-safe in
                # test_mtais_adapter.py.)
                dense = four_hot(rows, lat_bins, lon_bins, sog_bins, cog_bins)   # [T, D]
                targets_np = dense[:, np.newaxis, :]                              # [T, 1, D]
                inputs_np = np.zeros_like(targets_np)                             # pad-shift (vendor
                inputs_np[1:] = targets_np[:-1]                                   # process_AIS_batch)
                ts_np = sess.run(track_sample, feed_dict={
                    inputs_ph: inputs_np, targets_ph: targets_np,
                    lengths_ph: np.array([T], np.int32),
                    gs_ph: n_pre, ge_ph: T})
                samp = ts_np[n_pre:T]                                             # [n_held, S, D]
                bl = samp[:, :, :lat_bins].argmax(axis=2)
                bo = samp[:, :, lat_bins:lat_bins + lon_bins].argmax(axis=2)
                preds["g%d" % e["gap_id"]] = np.stack([bl, bo], axis=1).astype(np.int16)
                n_done += 1
                if n_done % 50 == 0:
                    print("[mtais-tf1] reconstructed %d/%d gaps (%.0fs)"
                          % (n_done, len(entries), time.time() - t0), flush=True)

    tmp = os.path.join(out_dir, "predictions.npz.tmp.npz")
    np.savez_compressed(tmp, **preds)
    final = os.path.join(out_dir, "predictions.npz")
    if os.path.exists(final):
        os.remove(final)
    os.rename(tmp, final)
    print("[mtais-tf1] reconstruction done: %d gaps (%d skipped) in %.0fs -> %s"
          % (n_done, n_skip, time.time() - t0, final), flush=True)


# ------------------------------------------------------------------------------------------------ #
def main():
    ap = argparse.ArgumentParser(description="MultitaskAIS stage-2 (TF1.15 CPU, py3.7)")
    ap.add_argument("--export-dir", help="absolute path to the stage-1 export for this cell")
    ap.add_argument("--out-dir", help="absolute path to the seed output dir")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--max-steps", type=int, default=30000)
    ap.add_argument("--batch-size", type=int, default=32)      # vendor README training invocation
    ap.add_argument("--latent-size", type=int, default=100)    # vendor README training invocation
    ap.add_argument("--num-samples", type=int, default=16)     # vendor traj_reconstruction
    ap.add_argument("--lr", type=float, default=0.0003)        # vendor default
    ap.add_argument("--test-cap", type=int, default=0, help="smoke: reconstruct first N gaps only")
    ap.add_argument("--selftest-load", default="",
                    help="load the given pickle with the numpy-compat shim, print a checksum, exit")
    a = ap.parse_args()

    if a.selftest_load:
        obj = _np_compat_load(a.selftest_load)
        if isinstance(obj, dict):
            ks = sorted(obj.keys())
            tot = sum(np.asarray(obj[k], np.float64).sum() for k in ks)
            shp = list(np.asarray(obj[ks[0]]).shape) if ks else []
            print("SELFTEST dict n=%d first_shape=%s checksum=%.6f" % (len(ks), shp, tot))
        else:
            arr = np.asarray(obj, np.float64)
            print("SELFTEST array shape=%s checksum=%.6f" % (list(arr.shape), arr.sum()))
        return

    if not (a.export_dir and a.out_dir):
        ap.error("--export-dir and --out-dir are required")
    if not os.path.isdir(a.out_dir):
        os.makedirs(a.out_dir)

    pred_path = os.path.join(a.out_dir, "predictions.npz")
    if os.path.exists(pred_path):
        print("[mtais-tf1] predictions.npz exists -- nothing to do (resume)", flush=True)
        return

    tm_path = os.path.join(a.out_dir, "train_meta.json")
    trained = False
    if os.path.exists(tm_path):
        try:
            with open(tm_path) as f:
                trained = bool(json.load(f).get("completed"))
        except Exception:
            trained = False
    if trained and tf.train.latest_checkpoint(os.path.join(a.out_dir, "chkpt")):
        print("[mtais-tf1] completed checkpoint found -- skipping training (resume)", flush=True)
    else:
        run_train(a.export_dir, a.out_dir, a.seed, a.epochs, a.max_steps,
                  a.batch_size, a.latent_size, a.lr)
    run_reconstruct(a.export_dir, a.out_dir, a.seed, a.num_samples, a.latent_size, a.test_cap)


if __name__ == "__main__":
    main()
