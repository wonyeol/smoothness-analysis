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
 ** apron_util.ml: parameterization with respect to Apron and utilities *)
open Apron
open Apron_sig

(** Available managers *)
module PA_box =
  (struct
    let module_name = "nd_PA_box"
    type t = Box.t
    let man: t Manager.t =
      Box.manager_alloc ()
  end: APRON_MGR)
module PA_oct =
  (struct
    let module_name = "nd_PA_oct"
    type t = Oct.t
    let man: t Manager.t =
      Oct.manager_alloc ()
  end: APRON_MGR)
module PA_pol =
  (struct
    let module_name = "nd_PA_polka"
    type t = Polka.strict Polka.t
    let man: t Manager.t =
      Polka.manager_alloc_strict ()
  end: APRON_MGR)

(** Utilities to use Apron APIs *)
(* Creation of Apron variables *)
let make_apron_var (id: string): Var.t =
  Var.of_string (Printf.sprintf "%s" id)
(* Apron constraint extraction *)
let extract_coeff_from_cons v cons =
  try Lincons1.get_coeff cons v
  with exn -> failwith "extract_coeff_from_cons"
(* Coefficient extraction *)
(* Scalar.t --> float --> int *)
let rec scalar_to_float (round: Mpfr.round) (s: Scalar.t) : float =
  (* Mpfr.round = Near | Zero | Up | Down *)
  match s with
  | Scalar.Float f -> f
  | Scalar.Mpfrf m -> Mpfrf.to_float ~round:round m
  | Scalar.Mpqf m -> scalar_to_float round (Scalar.Mpfrf (Mpfrf.of_mpq m round))

let float_to_int (round: Mpfr.round) (f: float): int =
  match round with
  | Mpfr.Up   -> int_of_float (ceil  f)
  | Mpfr.Down -> int_of_float (floor f)
  | _ -> failwith "float_to_int: unimplemented case"

let scalar_to_int (round: Mpfr.round) (s: Scalar.t): int =
  float_to_int round (scalar_to_float round s)

(** Pretty-printing *)
(* Basic utilities *)
let coeff_2str (c: Coeff.t): string =
  match c with
  | Coeff.Scalar scal -> Scalar.to_string scal
  | Coeff.Interval _ -> failwith "pp_coeff-interval"
let cons_trailer_2str (typ: Lincons0.typ): string =
  match typ with
  | Lincons1.EQ    -> " = 0"
  | Lincons1.DISEQ -> " != 0"
  | Lincons1.SUP   -> " > 0"
  | Lincons1.SUPEQ -> " >= 0"
  | Lincons1.EQMOD s -> Printf.sprintf " == 0 (%s)" (Scalar.to_string s)

(* Pretty-printing of an array of Apron constraints *)
let buf_linconsarray (ind: string) (buf: Buffer.t) (a: Lincons1.earray): unit =
  (* extraction of the integer variables *)
  let env = a.Lincons1.array_env in
  let ivars, fvars = Environment.vars env in
  (* pretty-printing of a constraint *)
  let f_lincons (cons: Lincons1.t): unit =
    (*Printf.bprintf buf "[%d,%d]" (Array.length ivars) (Array.length fvars);*)
    if Lincons1.is_unsat cons then Printf.bprintf buf "%sUNSAT\n" ind
    else
      (* print non zero coefficients *)
      let mt = ref false in
      Printf.bprintf buf "%s" ind;
      Array.iter
        (fun v ->
          let c = extract_coeff_from_cons v cons in
          if not (Coeff.is_zero c) then
            let vname = Var.to_string v in
            if !mt then Printf.bprintf buf " + "
            else mt := true;
            Printf.bprintf buf "%s . %s" (coeff_2str c) vname
        ) (Array.append fvars ivars);
      (* print the constant *)
      let d0 = coeff_2str (Lincons1.get_cst cons) in
      Printf.bprintf buf "%s%s" (if !mt then " + " else "") d0;
      (* print the relation *)
      Printf.bprintf buf "%s\n" (cons_trailer_2str (Lincons1.get_typ cons)) in
  (* Array of cons1 *)
  let ac1 =
    Array.mapi
      (fun i _ -> Lincons1.array_get a i)
      a.Lincons1.lincons0_array in
  Array.iter f_lincons ac1

