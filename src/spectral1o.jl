module spectral1o

using Printf
using DelimitedFiles
using DataInterpolations
using SpecialFunctions
using QuadGK

include("util.jl")
export writedat, filter_monotonic
include("eos.jl")
export load_crust, GammaCheb, get_eos

end
