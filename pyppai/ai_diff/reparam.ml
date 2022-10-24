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
 ** reparam.ml: reparameterisation transformation *)
open Analysis_sig
open Ir_sig
open Ir_util
open Lib


let sel_repar (k: SS.t) (p: prog): prog =
  let do_atomic acmd =
    match acmd with
    | Assert _
    | Assume _
    | Assn _
    | AssnCall _ -> acmd
    | Sample (idtf, n, d, el, obs_opt, repar) ->
        let parname =
          let dbg = false in
          match n with
          | Str s -> s
          | StrFmt (s, _) ->
              if dbg then
                F.printf "TODO,Sample formatter: %S\n" s;
              s
          | _ ->
              if dbg then
                F.printf "unbound parameter expression: %a\n" fp_expr n;
              failwith "unbound" in
        match obs_opt, repar with
        | Some _, _ -> acmd (* observe does not get reparameterised *)
        | _, true  -> acmd (* no double reparameterisation *)
        | None, false ->
            if false then F.printf "Reparam test for %s => %b\n" parname (SS.mem parname k);
            if SS.mem parname k then
              Sample (idtf, n, d, el, None, true)
            else acmd in
  let rec do_stmt stmt =
    match stmt with
    | Atomic acmd -> Atomic (do_atomic acmd)
    | If (e, b0, b1) -> If (e, do_block b0, do_block b1)
    | For (e0, e1, b) -> For (e0, e1, do_block b)
    | While (e, b) -> While (e, do_block b)
    | With (wil, b) -> With (wil, do_block b)
    | Break
    | Continue -> stmt
  and do_block b = List.map do_stmt b in
  do_block p
