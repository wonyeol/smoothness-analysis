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
 ** ddom_sig.ml: domains signatures for continuity/differentiability analysis *)
open Lib
open Ir_sig

open Diff_util

type constr = C_Pos | C_Neg | C_Num

let pp_constr chan = function
  | C_Pos -> Printf.printf "Pos"
  | C_Neg -> Printf.printf "Neg"
  | C_Num -> Printf.printf "Num" (* means numbers or contains of numbers *)
let fp_constr fmt = function
  | C_Pos -> F.fprintf fmt "Pos"
  | C_Neg -> F.fprintf fmt "Neg"
  | C_Num -> F.fprintf fmt "Num" (* means numbers or contains of numbers *)

type diff_info =
    { di_dens_diff:  SS.t ;
      di_dens_ndiff: SS.t ;
      di_prb_ndiff:  SS.t SM.t ;
      di_val_ndiff:  SS.t SM.t }

module type DOM_NUM =
  sig
    val name: string
    (* Set of parameters *)
    val init_domain: vars: SS.t -> unit
    (* Abstraction of the numerical state (variables -> values) *)
    type t
    (* Prtty-printing *)
    val pp: string -> out_channel -> t -> unit
    val fp: string -> form -> t -> unit
    (* Bottom check: when returns true, definitely bottom *)
    val is_bot: t -> bool
    (* Lattice operations *)
    val top: unit -> t
    val join: t -> t -> t
    val equal: t -> t -> bool
    (* Post-condition for assignments *)
    val forget: string -> t -> t
    val assign: string -> expr -> t -> t
    val heavoc: string -> constr -> t -> t
    (* Condition tests *)
    val guard: expr -> t -> t
    (* Operation on primitive-function call x=f(el) *)
    val call_prim: string (* x *) -> string (* f *)
      -> expr list (* el *) -> t -> t
    (* Operation on object call x=(c())(el) where c is an object constructor *)
    val call_obj: string (* x *) -> string (* c *)
      -> expr list (* el *) -> t -> t
    (* Operations on distributions *)
    val sample: string -> dist -> t -> t
    val check_dist_pars: dist_kind -> expr list -> t -> bool list option
    (* Approximate implication check. false means unknown *)
    val imply: t -> expr -> bool
  end

(** Module signature to represent information computed forward *)
module type DOM_DIFF_FWD =
  sig
    (* Abstractions of transformations *)
    type t
    (* Set of parameters *)
    val init_domain: goal:diff_prop -> unit
    (* Pretty-printing *)
    val fp: string -> form -> t -> unit
    val fp_results: string -> form -> t -> unit
    (* Accessing results *)
    val get_d_ndiff: t -> SS.t
    val get_d_diff:  SS.t -> t -> SS.t
    (* Lattice operations *)
    val join: t -> t -> t
    val equal: t -> t -> bool
    (* Abstraction of id *)
    val id: t
    (* Parameters *)
    val pars_add: string -> t -> t
    (* Guard parameters *)
    val guard_pars_get: t -> SS.t
    val guard_pars_set: SS.t -> t -> t
    val guard_pars_condition: expr -> t -> t
    (* Abstraction of basic operations *)
    val pyro_param: string    (* the variable *)
      -> string               (* the parameter *)
        -> t -> t
    val pyro_module: string     (* the variable *)
      -> (string * bool)        (* the parameter+whether it is differentiable *)
        -> t -> t
    val assign:
        ?safe_no_div0:(expr -> bool)
      -> string
        -> expr
          -> t -> t
    val call:
        ?safe_no_div0:(int -> expr -> bool)
      -> string * string * expr list (* x=f(el) *)
        -> (bool * bool list option)  (* diff, parameters diff *)
          -> t -> t
    val pyro_sample:
        ?safe_no_div0:(int option -> expr -> bool)
      -> ((dist_kind * dist_trans list) * bool * bool list)
        -> (string * expr * expr list * string)
          -> bool list option
            -> t -> t
    val pyro_observe:
        bool * bool list
      -> expr list * expr * bool list option
        -> t -> t
    val check_pyro_with:
        (expr * expr option) list
      -> t -> unit
    val check_no_dep:
        expr
      -> t -> unit
  end

(** Module signature to represent information computed compositionally *)
module type DOM_DIFF_COMP =
  sig
    val name: unit -> string
    val isnontop: bool
    (* Set of parameters *)
    val init_domain: goal:diff_prop -> params: SS.t -> vars: SS.t -> unit
    (* Abstractions of transformations *)
    type t
    (* Temporary *)
    val error: string -> t
    (* Prtty-printing *)
    val fp: string -> form -> t -> unit
    (* Abstraction of identity function *)
    val id: unit -> t
    (* Composition *)
    val compose: t -> t -> t
    (* Lattice operations *)
    val join: t -> t -> t
    val equal: t -> t -> bool
    (* Abstraction of basic operations *)
    val assign:
        ?safe:bool              (* whether the statement was proved safe *)
      -> idtf -> expr -> t
    val call: string            (* function name *)
      -> (bool * bool list)     (* diff signature: (diff,arguments diff) *)
        -> idtf option          (* output if any *)
          -> expr * expr list   (* function expression, arguments list *)
            -> t
    val pyro_param: string    (* the variable *)
      -> string               (* the parameter *)
        -> t
    val pyro_module: string     (* the variable *)
      -> (string * bool)        (* the parameter+whether it is differentiable *)
        -> t
    val pyro_sample: (* Pyro sample statement *)
        ?safei:((int -> bool)   (* safety information *)
                  * (int -> bool) * bool)
        -> repar:bool           (* whether it is a reparameterised sample *)
          -> string             (* the variable *)
            -> string           (* parameter *)
              -> (bool * bool list) (* distribution differentiability signature *)
                -> (expr * expr list) (* perameters of the sampling *)
                  -> t
    val pyro_observe: (* Pyro observe statement *)
        ?safei:((int -> bool)   (* safety information *)
                  * (int -> bool) * bool)
      -> (bool * bool list)     (* distribution differentiability signature *)
        -> expr list            (* distribution arguments *)
          -> expr
            -> t
    val loop_condition: (* Loop condition test: "V component" *)
        SS.t                  (* variables in the guard *)
      -> t                    (* body abstraction *)
        -> t
    val condition: (* Condition test effect on dependences/differentiability *)
        SS.t                  (* variables in the guard *)
      -> t                    (* body abstraction *)
        -> t
    (* Get differentiability information for density (diff,non diff) *)
    val get_density_diff_info: t -> diff_info
  end
