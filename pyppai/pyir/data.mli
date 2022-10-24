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
 ** data.ml: data programs to be used for batch testing (POPL'20) *)
open Analysis_sig

(** Expected outcome of a relational test case wrt. a default test *)
type test_oracle_r =
  | TOR_succeed (* the default test should succeed *)
  | TOR_fail    (* the default test should fail *)
  | TOR_error   (* must error *)
val string_of_test_oracle_r: test_oracle_r -> string

(** Default options *)
val aopts_default: analysis_opts

(** Description of tests *)
type use_zone = Zone | NoZone
type test_descr =
    { td_name:   string ;
      td_model:  string ;
      td_guide:  string ;
      td_zone:   use_zone ;
      td_result: test_oracle_r }

(** Options *)
val pyro_test_suite: test_descr list
val pyro_examples:   test_descr list
val suite:    (string * analysis_opts * analysis_opts * test_oracle_r) list
val examples: (string * analysis_opts * analysis_opts * test_oracle_r) list
