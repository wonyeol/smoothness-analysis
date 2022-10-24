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
 ** Copyright (c) 2019 KAIST and INRIA Paris
 **
 ** ai_diff.mli: entry point for continuity/differentiability analysis *)
open Analysis_sig
open Lib

open Ddom_sig

open Diff_util

(** Differentiability-related properties. *)
(* order on diff_prop *)
val diff_prop_leq: diff_prop -> diff_prop -> bool

(** Analysis main function wrapper.
 ** Inputs:
 ** - a domain.
 ** - a differentiability-related property `dp` to analyse.
 ** - a flag to activate or not second, compoisitional analysis
 ** - a file name.
 ** - a flag for verbose print.
 ** Outputs:
 ** - set of parameters w.r.t which density is `dp`.
 ** - set of parameters w.r.t which density may not be `dp`. *)
val analyze: ad_num -> diff_prop
  -> title: string (* title of the analysis (guide/model/...) *)
    -> flag_fwd: bool
      -> flag_comp:bool
        -> flag_old_comp:bool
          -> flag_pyast:bool
            -> inline_fns:string list
              -> input:(string,string * Ir_sig.prog) Either.t (* inpupt *)
                -> bool
                  -> diff_info
