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
 ** ir_parse.mli: entry point for continuity/differentiability analysis *)
open Analysis_sig

(** Output *)
val output: bool ref

(** Parsing *)
val parse_code: use_pyast:bool -> inline_fns:string list -> int option -> analysis_input -> Ir_sig.prog
