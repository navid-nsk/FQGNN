-- Library root of the FQ-GNN formalization: imports every module.
-- Common: the right-inverse pseudoinverse toolkit.
import FQGNN.Common.Pseudoinverse
-- Backbone: generator chart, unitary coupling, secular polynomial, torus lift, spectrum module.
import FQGNN.Backbone.Chart
import FQGNN.Backbone.Unitary
import FQGNN.Backbone.Secular
import FQGNN.Backbone.TorusEmbedding
import FQGNN.Backbone.SpectrumModule
-- Readout: exponential-sum comb, Op-H, FQ kernel / kriging, Op-M, blind-spot trade.
import FQGNN.Readout.ExponentialSum
import FQGNN.Readout.OpH
import FQGNN.Readout.Kernel
import FQGNN.Readout.OpM
import FQGNN.Readout.BlindSpot
-- Genuine Moore–Penrose inverse and the rank-free forms of Op-H-W / Op-M; the totally positive
-- periodic generator; thinned-spectrum rank; chart limit; value-block genericity;
-- infinite-spectrum kernel; Kronecker density.
import FQGNN.Common.MoorePenrose
import FQGNN.Readout.OpHGeneral
import FQGNN.Backbone.AperiodicityTP
import FQGNN.Backbone.SpectrumRankThinned
import FQGNN.Backbone.ChartLimit
import FQGNN.Readout.Genericity
import FQGNN.Readout.KernelInfinite
import FQGNN.Backbone.Kronecker
