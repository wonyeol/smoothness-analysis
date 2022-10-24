(** pyppai: basic abstract interpreter for python probabilistic programs
 **
 ** GNU General Public License
 **
 ** Authors:
 **  Wonyeol Lee, KAIST
 **  Xavier Rival, INRIA Paris
 **  Hongseok Yang, KAIST
 **  Hangyeol Yu, KAIST
 **
 ** Copyright (c) 2021 KAIST and INRIA Paris
 **
 ** ddom_diff.mli: domains signatures for differentiability information *)
open Lib

open Ddom_sig

(** Extracting results *)
val diff_info_fpi: string -> form -> diff_info -> unit
val diff_info_gen_dens_ndiff: diff_info -> SS.t
val diff_info_gen_dens_diff:  diff_info -> SS.t

(** Forward abstract domains **)
(* Forward abstract domain preserving no information
 *  only the set of parameters is computed  *)
module FD_none:     DOM_DIFF_FWD
(* Standard forward abstraction *)
module FD_standard: DOM_DIFF_FWD

(** Compositional abstract domains *)
(* No compositional abstraction (this domain is just one point) *)
module CD_none:    DOM_DIFF_COMP
(* Full compositional abstraction:
 *  with a representation that tracks variables/parameters with respect to which
 *  differentiability holds for sure *)
module CD_diff:    DOM_DIFF_COMP
(* Full compositional abstraction:
 *  with a representation that tracks variables/parameters with respect to which
 *  differentiability MAY NOT hold *)
module CD_ndiff:   DOM_DIFF_COMP
