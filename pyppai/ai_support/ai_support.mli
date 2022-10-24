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
 ** main.ml: launching of the analysis (with options already parsed) *)
open Ir_sig
open Analysis_sig
open Data

(** Master functions: *)
val start: bool -> analysis_opts list -> expr list -> expr list -> bool
val start_nr: bool -> analysis_opts -> expr list -> expr list -> bool
val start_r: bool -> analysis_opts -> analysis_opts -> test_oracle_r
  -> bool * string
val run_r: bool -> analysis_opts -> analysis_opts -> test_oracle_r
  -> bool option * bool
