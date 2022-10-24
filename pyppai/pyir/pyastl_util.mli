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
 ** pyastl_util.mli: utilities on Python local AST *)
open Pyastl_sig

(** Checking various properties of statements *)
val contains_continue: 'a stmt -> bool
val contains_break: 'a stmt -> bool
val contains_return: 'a stmt -> bool
val contains_middle_return: 'a stmt -> bool

(** Manipulate modules *)
val inline_funcdef: string list -> 'a modl -> 'a modl
                                         
(*
(** Conversion to strings *)
val string_of_number:     number       -> string
val string_of_boolop:     boolop       -> string
val string_of_operator:   operator     -> string
val string_of_unaryop:    unaryop      -> string
val string_of_cmpop:      cmpop        -> string
 *)
