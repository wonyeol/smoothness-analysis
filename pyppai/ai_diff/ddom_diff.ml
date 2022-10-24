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
 ** ddom_DIFF.ml: domains signatures for compositional information *)
open Lib

open Ddom_sig
open Ir_sig

open Diff_util
open Ir_util

(** Some configuration flags *)
let do_sanity_checks = false (* saves a couple of seconds *)


(** Extracting results *)
let diff_info_fpi (ind: string) (fmt: form) (di: diff_info): unit =
  let nind = "  "^ind in
  F.fprintf fmt "%sDens-Ndiff: %a\n%sPrb-Ndiff\n" ind ss_fp di.di_dens_ndiff ind;
  SM.iter (fun v s -> F.fprintf fmt "%s%s => %a\n" nind v ss_fp s) di.di_prb_ndiff;
  F.fprintf fmt "%sVal-Ndiff:\n" ind;
  SM.iter (fun v s -> F.fprintf fmt "%s%s => %a\n" nind v ss_fp s) di.di_prb_ndiff
let diff_info_gen_dens_ndiff (di: diff_info): SS.t =
  SM.fold (fun _ -> SS.union) di.di_prb_ndiff di.di_dens_ndiff
let diff_info_gen_dens_diff (di: diff_info): SS.t =
  SM.fold (fun _ s a -> SS.diff a s) di.di_prb_ndiff di.di_dens_diff


(** Forward abstract domains **)

(** Forward abstract domain preserving no information
 **  only the set of parameters is computed  *)
module FD_none =
  (struct
    (* Abstractions of transformations *)
    type t =
        { (* "Parameters":
           * Parameters with respect to which we track differentiability *)
          t_pars:    SS.t }
    (* Initialisation *)
    let init_domain ~(goal: diff_prop) = ()
    (* Prtty-printing *)
    let fp (ind: string) (fmt: form) (t: t): unit =
      F.fprintf fmt "%spars: %a\n" ind ss_fp t.t_pars
    let fp_results (ind: string) (fmt: form) (t: t): unit =
      F.fprintf fmt "%sParams found:\n\t%a\n" ind ss_fp t.t_pars
    (* Accessing results *)
    let get_d_ndiff (t: t): SS.t = t.t_pars
    let get_d_diff (allpars: SS.t) (t: t): SS.t = SS.empty
    (* Lattice operations *)
    let join (acc0: t) (acc1: t): t =
      { t_pars    = SS.union      acc0.t_pars    acc1.t_pars }
    let equal (acc0: t) (acc1: t): bool =
      SS.equal            acc0.t_pars    acc1.t_pars
    (* Abstraction of id *)
    let id = { t_pars = SS.empty }

    (* Parameters *)
    let pars_add (pname: string) (t: t): t =
      { t_pars = SS.add pname t.t_pars }

    (* Guard parameters *)
    let guard_pars_get (t: t): SS.t = SS.empty
    let guard_pars_set (_: SS.t) (t: t): t = t
    let guard_pars_condition (e: expr) (t: t): t = t

    (* Transfer functions *)
    let pyro_param _x pname t =
      { t_pars    = SS.add pname t.t_pars }
    let pyro_module x (pname, _isdiff) t =
      { t_pars    = SS.add pname t.t_pars }
    let call
        ?(safe_no_div0: int -> expr -> bool = fun _ _ -> false)
        ((x, v, el): string * string * expr list)
        (is_ipdiff,ofsig) (t: t): t =
      t
    let assign
        ?(safe_no_div0: expr -> bool = fun _ -> false)
        x e (t: t): t =
      t
    let pyro_sample
        ?(safe_no_div0: int option -> expr -> bool = fun _ _ -> false)
        (d, dist_diff, fsig_diff)
        (x, n, a, parname)
        sel
        (t: t): t =
      { t_pars    = SS.add parname t.t_pars }
    let pyro_observe
        (dist_diff, fsig_diff)
        (a, o, sel)
        (t: t): t =
      t
    let check_pyro_with (l: (expr * expr option) list) (t: t): unit =
      ( )
    let check_no_dep (e: expr) (t: t): unit =
      ( )
  end: DOM_DIFF_FWD)

(** Standard forward abstraction **)
module FD_standard =
  (struct
    let r_goal = ref Top
    (* Abstractions of transformations *)
    type t =
        { (* "Parameters":
           * Parameters with respect to which we track differentiability *)
          t_pars:    SS.t ;
          (* "Guard Parameters"
           * Parameters that are may guard the current path;
           * => we cannot guarantee anything is differentiable wrt those *)
          t_gpars:   SS.t ;
          (* "Variables-Parameter-Non-Diff":
           * Maps each variable to parameters with respect to which it may not
           * be differentiable. If a variable x stores a function f that may
           * depend on parameters, we say that it is differentiable wrt.
           * p if f(v) is differentiable wrt. p for all v.
           * (* WL: Need the last line? In all other parts, we implicitly
           *  * assume that
           *  * "f : X x Y -> R is differentiable wrt x" means
           *  * "f(_,y) is differentiable wrt x for all y",
           *  * which is exactly the same as what the last line says. *) *)
          t_vpndiff: SS.t SM.t ;
          (* "Variables-Parameter-Dependencies":
           * Maps each variable to parameters that it may depend on *)
          t_vpdep:   SS.t SM.t ;
          (* "Density-Parameter-Non-Diff":
           * Set of parameters for which density may not be differentiable *)
          t_dpndiff: SS.t }
    (* Initialisation *)
    let init_domain ~(goal: diff_prop): unit = r_goal := goal
    (* Prtty-printing *)
    let fp (ind: string) (fmt: form) (t: t): unit =
      let fpm_ss = fpm ss_fp in
      F.fprintf fmt "%spars: %a\n%spnd:\n%a%sdepd:\n%a%sgpars: %a\n"
        ind ss_fp t.t_pars
        ind fpm_ss t.t_vpndiff
        ind fpm_ss t.t_vpdep
        ind ss_fp  t.t_gpars;
      F.fprintf fmt "%sdndiff: %a\n"
        ind ss_fp  t.t_dpndiff
    let fp_results (ind: string) (fmt: form) (t: t): unit =
      F.fprintf fmt "%sParams found:\n\t%a\n" ind ss_fp t.t_pars;
      F.fprintf fmt "%s Non-differentiability of variables wrt parameters:\n" ind;
      SM.iter
        (fun v pars ->
          F.fprintf fmt "\t%-30s\t=>\t%a\n" v ss_fp pars
        ) t.t_vpndiff;
      F.fprintf fmt "%sNon-differentiability of density wrt parameters:\n\t%a\n"
        ind ss_fp t.t_dpndiff
    (* Accessing results *)
    let get_d_ndiff (t: t): SS.t = t.t_dpndiff
    let get_d_diff (allpars: SS.t) (t: t): SS.t = SS.diff allpars t.t_dpndiff
    (* Lattice operations *)
    let join (acc0: t) (acc1: t): t =
      let ss_map_join m0 m1 =
        map_join_union SS.union m0 m1 in
      { t_pars    = SS.union      acc0.t_pars    acc1.t_pars;
        t_gpars   = SS.union      acc0.t_gpars   acc1.t_gpars;
        t_vpndiff = ss_map_join   acc0.t_vpndiff acc1.t_vpndiff;
        t_vpdep   = ss_map_join   acc0.t_vpdep   acc1.t_vpdep;
        t_dpndiff = SS.union      acc0.t_dpndiff acc1.t_dpndiff; }
    let equal (acc0: t) (acc1: t): bool =
      let ss_map_equal m0 m1 =
        map_equal (fun v ss -> ss <> SS.empty) SS.equal m0 m1 in
      SS.equal            acc0.t_pars    acc1.t_pars
        && SS.equal       acc0.t_gpars   acc1.t_gpars
        && ss_map_equal   acc0.t_vpndiff acc1.t_vpndiff
        && ss_map_equal   acc0.t_vpdep   acc1.t_vpdep
        && SS.equal       acc0.t_dpndiff acc1.t_dpndiff
    (* Abstraction of id *)
    let id =
      { t_pars    = SS.empty ;
        t_vpndiff = SM.empty ;
        t_vpdep   = SM.empty ;
        t_gpars   = SS.empty ;
        t_dpndiff = SS.empty ; }
    (** Utilities *)
    type texp =
        { (* Parameters on which the expression may depend on *)
          te_pdep:   SS.t ;
          (* Parameters with respect to which the expression may depend on
           * and may be non-differentiable:
           * => in general it is always sound to make this field equal to
           *    te_pdep; *)
          te_pndiff: SS.t }
    let accumulate_guard_pars (accu: SS.t) (del: texp list)
        (ok_el: bool list option): SS.t =
      let default () =
        List.fold_left (fun acc de -> SS.union acc de.te_pdep) accu del in
      match ok_el with
      | None -> default ()
      | Some ok_el ->
          try
            let f acc ok de = if ok then acc else SS.union acc de.te_pdep in
            List.fold_left2 f accu ok_el del
          with Invalid_argument _ -> default ()
    (* Compute differentiability information for expressions *)
    let diff_expr
        (safe_no_div0: expr -> bool)
        (t: t) (e: expr): texp =
      let rec aux = function
        | Nil | True | False | Ellipsis | Num _ | Str _ ->
            { te_pdep   = SS.empty ;
              te_pndiff = SS.empty }
        | Name x ->
            let pdep   =
              try SM.find x t.t_vpdep   with Not_found -> SS.empty in
            let pndiff =
              try SM.find x t.t_vpndiff with Not_found -> SS.empty in
            { te_pdep   = pdep ;
              te_pndiff = pndiff }
        | UOp (Not, e) ->
            let te = aux e in
            { te with
              te_pndiff = te.te_pdep }
        | BOp ((Add | Sub | Mult | Pow), e0, e1) ->
            (* numeric, differentiable cases *)
            let te0 = aux e0 and te1 = aux e1 in
            { te_pdep   = SS.union te0.te_pdep   te1.te_pdep ;
              te_pndiff = SS.union te0.te_pndiff te1.te_pndiff }
        | BOp (Div, e0, e1) ->
            (* numeric, partly differentiable:
             * - Div: discontinuity at 0 *)
            let te0 = aux e0 and te1 = aux e1 in
            let no_div0 =
              safe_no_div0 e1 in
            (*(SI.no_div0 (*!ref_*)safety_info loc)
              || imply (*acc*) (Comp (NotEq, e1, Num (Float 0.))) in*)
            let te_pdep = SS.union te0.te_pdep te1.te_pdep in
            (* TODO: XR: this is the sign that the state analysis helps;
             * => we need to compute a boolean stating that the state analysis
             *    succeeded and use it to improve the precision of the
             *    compositional analysis *)
            let te_pndiff =
              if no_div0 then SS.union te0.te_pndiff te1.te_pndiff
              else SS.union te0.te_pndiff te1.te_pdep in
            if false then
              F.printf "COMP,Checking if div exression is ok(%a): %b,%a\n"
                fp_expr e no_div0 ss_fp te_pndiff;
            { te_pdep   = te_pdep ;
              te_pndiff = te_pndiff }
        | BOp ((And | Or), e0, e1)
        | Comp (_, e0, e1) ->
            (* comparison and boolean operators *)
            let te0 = aux e0 and te1 = aux e1 in
            let pdep = SS.union te0.te_pdep te1.te_pdep in
            { te_pdep   = pdep ;
              te_pndiff = pdep }
        | UOp ((SampledStr | SampledStrFmt), e0) ->
            let te0 = aux e0 in
            { te0 with
              te_pndiff = te0.te_pdep }
        | List el ->
            let tei = { te_pdep   = SS.empty ;
                        te_pndiff = SS.empty } in
            List.fold_left
              (fun a ep ->
                let tep = aux ep in
                { te_pdep   = SS.union tep.te_pdep   a.te_pdep ;
                  te_pndiff = SS.union tep.te_pndiff a.te_pndiff }
              ) tei el
        | StrFmt (_, el) ->
            let pdep =
              List.fold_left (fun a te -> SS.union a te.te_pdep)
                SS.empty (List.map aux el) in
            { te_pdep   = pdep ;
              te_pndiff = pdep }
        | e ->
            F.printf "TODO expression: %a\n" fp_expr e;
            { te_pdep   = t.t_pars ;
              te_pndiff = t.t_pars } in
      aux e
    let ndpars_call_args (t: t) (fsig: bool list) (del: texp list): SS.t =
      let rec aux (fsig: bool list) (del: texp list): SS.t =
        match fsig, del with
        | diffarg :: fsig, d :: del ->
            let pn = aux fsig del in
            let pndiff =
              if diffarg then d.te_pndiff
              else d.te_pdep in
            SS.union pn pndiff
        | _ :: _, [ ] ->
            (* May not be differentiable at all *)
            t.t_pars
        | [ ], d :: del ->
            let pn = aux [ ] del in
            SS.union pn d.te_pdep
        | [ ], [ ] ->
            SS.empty in
      aux fsig del

    (* Parameters *)
    let pars_add (pname: string) (t: t): t =
      { t with t_pars = SS.add pname t.t_pars }

    (* Guard parameters *)
    let guard_pars_get (t: t): SS.t = t.t_gpars
    let guard_pars_set (gpars: SS.t) (t: t): t = { t with t_gpars = gpars }
    let guard_pars_condition (e: expr) (t: t): t =
      let temp _ = false in (* TODO!!!! *)
      let d = diff_expr temp t e in
      if false (*!debug*) then
        F.printf "CONDITION: %s\n"
          (if d.te_pdep = SS.empty then "precise" else "imprecise");
      { t with t_gpars = SS.union t.t_gpars d.te_pdep }

    let pyro_param x pname t =
      { t with
        t_pars    = SS.add pname t.t_pars;
        t_vpndiff = SM.add x SS.empty t.t_vpndiff;
        t_vpdep   = SM.add x (SS.singleton pname) t.t_vpdep }

    let pyro_module x (pname, isdiff) t =
      let pdep =
        let pdep_old = lookup_with_default x t.t_vpdep SS.empty in
        SS.add pname pdep_old in
      let pndiff =
        let pndiff_old = lookup_with_default x t.t_vpndiff SS.empty in
        if isdiff then pndiff_old
        else SS.add pname pndiff_old in
      { t with
        t_vpndiff = SM.add x pndiff t.t_vpndiff;
        t_pars    = SS.add pname t.t_pars;
        t_vpdep   = SM.add x pdep t.t_vpdep }

    let call
        ?(safe_no_div0: int -> expr -> bool = fun _ _ -> false)
        ((x, v, el): string * string * expr list)
        (is_ipdiff,ofsig) (t: t): t =
      if dbg_call then
        F.printf "Fwd,call %s = %s(...)\n" x v;
      let diff_expr i = diff_expr (fun e -> safe_no_div0 i e) in
      let del = List.mapi (fun i -> diff_expr i (*(Loc.parn loc i)*) t) el in
      (* All dependencies in the arguments *)
      let deppar_el =
        List.fold_left (fun a e -> SS.union a e.te_pdep) SS.empty del in
      let deppar_v = lookup_with_default v t.t_vpdep SS.empty in
      let deppar = SS.union deppar_v (SS.union deppar_el t.t_gpars) in
      (* Check if the return value is differentiable *)
      let pndiff =
        match is_ipdiff, ofsig with
        | true, Some fsig -> ndpars_call_args t fsig del
        | false, Some fsig -> SS.union deppar_v (ndpars_call_args t fsig del)
        | _, None -> deppar in
      (* Non-differentiability with respect to guard parameters *)
      let pndiff = SS.union pndiff t.t_gpars in
      let r = { t with
                t_vpndiff = SM.add x pndiff t.t_vpndiff;
                t_vpdep   = SM.add x deppar t.t_vpdep } in
      if dbg_call then
        F.printf "Fwd,Call:\n%a\n" (fp "  ") r;
      r

    let assign
        ?(safe_no_div0: expr -> bool = fun _ -> false)
        x e (t: t): t =
      (* Non-differentiability information about the RHS expression *)
      let de = diff_expr safe_no_div0 t e in
      (* Non-differentiability with respect to guard parameters *)
      let pdep   = SS.union de.te_pdep   t.t_gpars in
      let pndiff = SS.union de.te_pndiff t.t_gpars in
      { t with
        t_vpndiff = SM.add x pndiff t.t_vpndiff;
        t_vpdep   = SM.add x pdep t.t_vpdep }

    let pyro_sample
        ?(safe_no_div0: int option -> expr -> bool = fun _ _ -> false)
        (d, dist_diff, fsig_diff)
        (x, n, a, parname)
        sel
        (t: t): t =
      if dbg_sample then
        F.printf "Fwd,Sample %s,(%s)\n%a"
          (dist_kind_to_string (fst d)) parname (fp "  ") t;
      let del =
        List.mapi
          (fun i ->
            diff_expr (fun e -> safe_no_div0 (Some i) e) t
          ) a in
      let dn  = diff_expr (fun e -> safe_no_div0 None e) t n in
      if dn.te_pdep != SS.empty then
        F.printf "TODO: sample dist expr dep\n";
      (* Guarding *)
      let gpars =
        accumulate_guard_pars (SS.union t.t_gpars dn.te_pdep) del sel in
      (* Dependency of a sampled variable *)
      let deppar = SS.add parname gpars in
      (* Differentiability of the density *)
      let dndiff =
        let ndpars0 = ndpars_call_args t fsig_diff del in
        let ndpars1 =
          if dist_diff then ndpars0 else SS.add parname ndpars0 in
        F.printf "Fwd,Sample,middle: %b\n %a\n %a\n" dist_diff
          ss_fp ndpars0 ss_fp ndpars1;
        SS.union gpars ndpars1 in
      let r = { t_pars    = SS.add parname t.t_pars;
                t_gpars   = gpars;
                t_vpndiff = SM.add x gpars t.t_vpndiff;
                t_vpdep   = SM.add x deppar t.t_vpdep;
                t_dpndiff = SS.union dndiff t.t_dpndiff } in
      if dbg_sample then
        F.printf "Fwd,Sample (%s): %b, %a\nFwd,Sample: dp%a\n%a"
          (dist_kind_to_string (fst d)) dist_diff ss_fp dndiff
          fp_dist_par sel (fp "  ") r;
      r

    let pyro_observe
        (dist_diff, fsig_diff)
        (a, o, sel)
        (t: t): t =
      let safe_div0 = fun _ -> failwith "todo:observe:diff:ex" in
      let del =
        List.mapi (fun i -> diff_expr (*(Loc.parn loc i)*) safe_div0 t) a in
      let d_o = diff_expr (*loc*) safe_div0 t o in
      (* Guarding parameters *)
      let gpars = accumulate_guard_pars t.t_gpars del sel in
      (* Differentiability of the density *)
      let ndensdiff =
        (* TODO:
         * XR: I am a bit concerned about the next few lines;
         *     are we considering differentiability of dist density ?
         *)
        if dist_diff then
          SS.union d_o.te_pndiff (ndpars_call_args t fsig_diff del)
        else
          SS.union d_o.te_pdep (ndpars_call_args t fsig_diff del) in
      if dbg_observe then
        F.printf "Fwd,Observe: %b, %a\nFwd,Observe: dp=%a\n"
          dist_diff ss_fp ndensdiff fp_dist_par sel;
      { t with
        t_gpars   = gpars;
        t_dpndiff = SS.union ndensdiff t.t_dpndiff }

    let check_pyro_with (l: (expr * expr option) list) (t: t): unit =
      (* If the with items use any differentiation parameter, we give up *)
      let b =
        let safe_div0 = fun _ -> failwith "todo:with:diff:ex" in
        List.fold_left
          (fun b -> function
            | e, None ->
                b && (diff_expr safe_div0 t e).te_pdep != SS.empty
            | e, Some o ->
                b && (diff_expr safe_div0 t e).te_pdep != SS.empty
                  && (diff_expr safe_div0 t o).te_pdep != SS.empty
          ) true l in
      (* Note this message should only be issued
       * in state checking analysis!!! *)
      if b then
        Printf.printf "TODO,with,expression depending on parameters\n"

    let check_no_dep (e: expr) (t: t): unit =
      let safe_no_div0 _ = false in
      let d = diff_expr safe_no_div0 t e in
      if d.te_pdep != SS.empty then failwith "TODO,for"
  end: DOM_DIFF_FWD)




(** Compositional abstract domains *)

(** No compositional abstraction (this domain is just one point) *)
module CD_none =
  (struct
    let name () = "single point compositional abstraction; always returns top"
    let isnontop = false
    (* Set of parameters *)
    let init_domain ~(goal: diff_prop) ~(params: SS.t) ~(vars: SS.t): unit = ()
    (* Abstractions of transformations *)
    type t = unit
    (* Prtty-printing *)
    let fp (ind: string) (_: form) (t: t): unit = ()
    (* Temporary *)
    let error (_: string): t = ()
    (* Abstraction of identity function *)
    let id (): t = ()
    (* Composition *)
    let compose (_: t) (_: t): t = ()
    (* Lattice operations *)
    let join (_: t) (_: t): t = ()
    let equal (_: t) (_: t): bool = true
    (* Abstraction of basic operations *)
    let assign ?(safe: bool = false) (_: idtf) (_: expr): t = ()
    let call _ _ _ _ = ()
    let pyro_param _ _ = ()
    let pyro_module _ _ = ()
    let ftop _ = false
    let pyro_sample
        ?(safei: (int -> bool) * (int -> bool) * bool = (ftop,ftop,false))
        ~(repar: bool)
        _ _ _ _ = ()
    let pyro_observe
        ?(safei: (int -> bool) * (int -> bool) * bool = (ftop,ftop,false))
        _ _ _  = ()
    let loop_condition _ _ = ()
    let condition _ _ = ()
    (* Get differentiability information for density (diff,non diff) *)
    let get_density_diff_info _ = { di_dens_diff  = SS.empty ;
                                    di_dens_ndiff = SS.empty ;
                                    di_prb_ndiff  = SM.empty ;
                                    di_val_ndiff  = SM.empty }
  end: DOM_DIFF_COMP)

(** Full compositional abstraction:
 **  with a representation that tracks variables/parameters with respect to which
 **  differentiability holds for sure *)
module CD_diff =
  (struct
    let r_goal = ref Top
    let name () = F.asprintf "compositional %a" fp_diff_prop !r_goal
    let isnontop = true
    (* Some utilities, put here for now, maybe move later *)
    type par =
      | PVar of string     (* Program variable *)
      | PParDens of string (* Density of random parameter *)
      | PDens              (* Density of execution (like) *)
    module POrd =
      struct
        type t = par
        let compare = compare
      end
    module PS = Set.Make( POrd )
    module PM = Map.Make( POrd )
    (* Abstractions of transformations *)
    type u =
        { (* modified parameters
           *)
          t_mod:  PS.t ;
          (* dependency partial map
           *    when x is not in the map, it means x=>{x}
           *    which means that x depends only on itself *)
          t_dep:  PS.t PM.t;
          (* differentiability partial map
           *    when x is not in the map, it meanx x=>V#
           *    which means x is differentiable wrt all variables *)
          t_diff: PS.t PM.t;
          (* possible (non)-termination dependencies
           *    variables on which the termination of the command may depend *)
          t_v:    PS.t }
    type t =
      | T_ok of u (* to fill *)
      | T_err of string
    (* Set of parameters *)
    let ref_all_pars: SS.t ref = ref SS.empty
    let ref_all_vars: SS.t ref = ref SS.empty
    let ref_all_ps:   PS.t ref = ref PS.empty
    let ref_uid: u ref = ref { t_mod  = PS.empty ;
                               t_dep  = PM.empty ;
                               t_diff = PM.empty;
                               t_v    = PS.empty }
    let ref_id: t ref =
      (* before initialisation *)
      ref (T_err "undefined init")
    let init_domain ~(goal: diff_prop) ~(params: SS.t) ~(vars: SS.t): unit =
      r_goal := goal;
      if dbg_init then
        F.printf "Domain init:\n - parameters: %a\n - vars: %a\n"
          ss_fp params ss_fp vars;
      ref_all_pars := params;
      ref_all_vars := vars;
      ref_all_ps :=
        SS.fold (fun x -> PS.add (PVar x)) !ref_all_vars
          (SS.fold (fun x -> PS.add (PParDens x)) !ref_all_pars
             (PS.singleton PDens));
      let uid =
        let dep =
          PS.fold (fun x -> PM.add x (PS.singleton x)) !ref_all_ps PM.empty in
        let diff =
          PS.fold (fun x -> PM.add x !ref_all_ps) !ref_all_ps PM.empty in
        { t_mod  = PS.empty;
          t_dep  = dep;
          t_diff = diff;
          t_v    = PS.empty } in
      ref_uid := uid;
      ref_id := T_ok uid
    (* Temporary *)
    let error (msg: string): t = T_err msg
    (* Prtty-printing *)
    let par_fp fmt = function
      | PVar v -> F.fprintf fmt "%s" v
      | PParDens v -> F.fprintf fmt "[%s:dens]" v
      | PDens -> F.fprintf fmt "<dens>"
    let ps_fp fmt s =
      F.fprintf fmt "{ ";
      PS.iter (F.fprintf fmt "%a; " par_fp) s;
      F.fprintf fmt "}"
    let fp (ind: string) (fmt: form) (t: t): unit =
      let subind = "    "^ind in
      let fs fmt s = ps_fp fmt s in
      let fm fmt m =
        PM.iter
          (fun x -> F.fprintf fmt "%s%a => %a\n" subind par_fp x fs)
          m in
      match t with
      | T_ok u ->
          let ndiff = PM.map (fun s -> PS.diff !ref_all_ps s) u.t_diff in
          F.fprintf fmt "%sOk state\n%s  Mods: %a\n%s  Deps:\n%a"
            ind ind fs u.t_mod ind fm u.t_dep;
          F.fprintf fmt "%s  May not %a:\n%a%s  V: %a\n"
            ind fp_diff_prop_short !r_goal fm ndiff ind fs u.t_v
      | T_err s -> F.fprintf fmt "%sKo state [ %s ]\n" ind s
    (* Some functions for debugging *)
    let crash msg =
      flush stdout;
      failwith msg
    let sanity_check (msg: string) (t: t): unit =
      let errors = ref [ ] in
      (* check well-formedness of a map to sets:
       *  - keys should be exactly the set of all dimensions
       *  - each set should be included in the set of dimensions *)
      let check (name: string) (m: PS.t PM.t): unit =
        let add_error msg =
          errors := (name, msg) :: !errors in
        PS.iter
          (fun i ->
            if not (PM.mem i m) then
              add_error (F.asprintf "missing par %a" par_fp i)
          ) !ref_all_ps;
        PM.iter
          (fun i s ->
            if not (PS.mem i !ref_all_ps) then
              add_error (F.asprintf "unbound map par %a" par_fp i);
            PS.iter
              (fun i ->
                if not (PS.mem i !ref_all_ps) then
                  add_error (F.asprintf "unbound set par %a" par_fp i)
              ) s
          ) m in
      let do_u (u: u): unit =
        (* modified variables should all be defined *)
        if not (PS.subset u.t_mod !ref_all_ps) then
          PS.iter
            (fun p ->
              let msg = F.asprintf "%a found" par_fp p in
              errors := ("modified pars", msg) :: !errors
            ) (PS.diff !ref_all_ps u.t_mod);
        (* maps of dependences and differentiability should be well-formed *)
        check "dep" u.t_dep;
        check "diff" u.t_diff;
        (* for each variable, non differentiable pars should also appear in
         * dependences *)
        PM.iter
          (fun i d ->
            let p = try PM.find i u.t_diff with Not_found -> PS.empty in
            let np = PS.diff !ref_all_ps p in
            if not (PS.subset np d) then
              errors := ("!(ndiff<=dep)", F.asprintf "%a" par_fp i) :: !errors
          ) u.t_dep;
        (* error report *)
        if !errors != [] then
          begin
            F.printf "Validity condition failed (%s):\n  All: %a\n" msg ps_fp
              !ref_all_ps;
            List.iter
              (fun (name, error) -> F.printf "  in %s: %s\n" name error)
              (List.rev !errors);
            F.printf "In abstract state:\n%a\n" (fp "  ") t
          end in
      if do_sanity_checks then
        match t with
        | T_ok u -> do_u u
        | T_err _ -> ()
    (* Abstraction of identity function *)
    let id (): t =
      let r = !ref_id in
      sanity_check "id" r;
      r
    (* Composition *)
    let compose (t0: t) (t1: t): t =
      sanity_check "compose-l" t0;
      sanity_check "compose-r" t1;
      (* TODO: issue: we need to require allpars here,
       *              which is not satisfactory *)
      match t0, t1 with
      | T_ok u0, T_ok u1 ->
          let get_dep0 x =
            try PM.find x u0.t_dep with Not_found -> PS.empty in
          let get_deps0 s =
            PS.fold (fun y -> PS.union (get_dep0 y)) s PS.empty in
          let get_diff0 x =
            try PM.find x u0.t_diff
            with Not_found -> crash "" (*(F.sprintf "get_diff: %s" x)*) in
          let get_diffs0 s =
            PS.fold (fun y -> PS.inter (get_diff0 y)) s !ref_all_ps in
          let dep =
            PM.fold
              (fun x d1 acc ->
                let d = PS.fold (fun y -> PS.union (get_dep0 y)) d1 u0.t_v in
                PM.add x d acc
              ) u1.t_dep PM.empty in
          let diff =
            PM.fold
              (fun x p1 acc ->
                let d1 = try PM.find x u1.t_dep with Not_found -> PS.empty in
                (*if not (PS.subset (PS.diff !ref_all_ps p1) d1) then
                  F.printf "WARN: inclusion: %a\n"
                    ps_fp (PS.diff (PS.diff !ref_all_ps p1) d1);*)
                let p = get_diffs0 (PS.inter p1 d1) in
                let f y = not (PS.mem y (get_deps0 (PS.diff d1 p1))) in
                let p = PS.filter f p in
                let p = PS.diff p u0.t_v in
                PM.add x p acc
              ) u1.t_diff PM.empty in
          let v =
            PS.fold
              (fun x acc ->
                try PS.union (PM.find x u0.t_dep) acc
                with Not_found -> acc
              ) u1.t_v u0.t_v in
          let r = T_ok { t_mod  = PS.union u0.t_mod u1.t_mod ;
                         t_dep  = dep ;
                         t_diff = diff ;
                         t_v    = v } in
          sanity_check "compose-out" r;
          if dbg_compose then
            F.printf "Compose:\n%a\n%a\n%a" (fp "     ") t0
              (fp "     ") t1 (fp "   ") r;
          r
      | T_ok _, _ -> t1
      | _, _ -> t0
    (* Lattice operations *)
    let join (t0: t) (t1: t): t =
      match t0, t1 with
      | T_ok u0, T_ok u1 ->
          let fm o m0 m1 =
            PM.fold
              (fun p s0 acc ->
                let s1 =
                  try PM.find p m1
                  with Not_found -> failwith "incorrect state" in
                PM.add p (o s0 s1) acc
              ) m0 PM.empty in
          T_ok { t_mod  = PS.union u0.t_mod u1.t_mod ;
                 t_dep  = fm PS.union u0.t_dep u1.t_dep ;
                 t_diff = fm PS.inter u0.t_diff u1.t_diff ;
                 t_v    = PS.union u0.t_v u1.t_v }
      | T_ok _, _ -> t1
      | _, _ -> t0
    let equal (t0: t) (t1: t): bool =
      try
        match t0, t1 with
        | T_ok u0, T_ok u1 ->
            if not (PS.equal u0.t_mod u1.t_mod) then raise Stop;
            let f m0 m1 =
              let g m = PM.fold (fun p _ -> PS.add p) m PS.empty in
              if not (PS.equal (g m0) (g m1)) then raise Stop;
              PM.iter
                (fun p s0 ->
                  let s1 = try PM.find p m1 with Not_found -> raise Stop in
                  if not (PS.equal s0 s1) then raise Stop) m0 in
            f u0.t_dep u1.t_dep;
            f u0.t_diff u1.t_diff;
            if not (PS.equal u0.t_v u1.t_v) then raise Stop;
            true
        | T_err _, T_err _ -> true
        | _, _ -> false
      with Stop -> false
    (* Abstraction of basic operations *)
    let vars_to_ps (s: SS.t): PS.t =
      SS.fold (fun x -> PS.add (PVar x)) s PS.empty
    let dep_expr_ps (e: expr) = vars_to_ps (dep_expr e)
    let rec diff_expr ?(safe: bool = false) (e: expr): PS.t =
      let rec aux_expr (acc: PS.t): expr -> PS.t = function
        | Nil | True | False | Num _ | Str _ | Name _ | Ellipsis -> acc
        | BOp ((Add | Sub | Mult | Pow), e0, e1) ->
            aux_expr (aux_expr acc e0) e1
        | BOp (Div, e0, e1) ->
            if safe then aux_expr (aux_expr acc e0) e1
            else PS.diff (aux_expr (aux_expr acc e0) e1) (dep_expr_ps e1)
        | Comp (_, e0, e1) ->
            PS.diff !ref_all_ps (PS.union (dep_expr_ps e0) (dep_expr_ps e1))
        | List el ->
            List.fold_left (fun acc e -> aux_expr acc e) acc el
        | _ ->
            F.printf "TODO[diff_expr]:\n  %a\n\n\n" fp_expr e; flush stdout;
            failwith "todo: diff_expr" in
      aux_expr !ref_all_ps e
    let assign ?(safe: bool = false) (x: idtf) (e: expr): t =
      let u = !ref_uid in
      let pvar = PVar x in
      let r =
        T_ok { t_mod  = PS.singleton pvar ;
               t_dep  = PM.add pvar (dep_expr_ps e) u.t_dep ;
               t_diff = PM.add pvar (diff_expr ~safe e) u.t_diff ;
               t_v    = PS.empty } in
      sanity_check "assign" r;
      r
    let call_diff_info ?(safei: int -> bool = fun _ -> false) c =
      let rec aux i diff = function
        | b :: fsig, de :: del, e :: eas ->
            (*let old = diff in*)
            let diff =
              if b then (* parameter is differentiable *)
                PS.inter (diff_expr ~safe:(safei i) e) diff
              else (* parameter not differentiable *)
                PS.diff diff de in
            (*F.printf "call_diff[%a]: %b => %a (%a)\n" fp_expr e b*)
            (*  ps_fp diff ps_fp old;*)
            aux (i+1) diff (fsig, del, eas)
        | [ ], [ ], [ ] -> diff
        | [ ], de :: del, _ :: eas ->
            aux (i+1) (PS.diff diff de) ([ ], del, eas)
        | _, _, _ -> failwith (F.asprintf "incoherent call: %s" (name ())) in
      (*F.printf "call_diff_starts: %a\n" ps_fp !ref_all_ps;*)
      aux 0 !ref_all_ps c
    let call
        (name: string)
        ((is_ipdiff,fsig): bool * bool list)
        (xo: idtf option) ((ef, eas): expr * expr list): t =
      assert is_ipdiff;
      let r =
        match xo with
        | None ->
            error "call proc, todo"
        | Some x ->
            let del = List.map dep_expr_ps eas in
            let dep = List.fold_left PS.union PS.empty del in
            let diff = call_diff_info (fsig, del, eas) in
            let u = !ref_uid in
            let var = PVar x in
            let u = { t_mod  = PS.singleton var ;
                      t_dep  = PM.add var dep u.t_dep ;
                      t_diff = PM.add var diff u.t_diff ;
                      t_v    = PS.empty } in
            T_ok u in
      if dbg_call then
        F.printf "call: %s\n%a" name (fp "  ") r;
      sanity_check "call" r;
      r
    let pyro_param (x: string) (sparam: string): t =
      let u = !ref_uid in
      let var = PVar x and param = PParDens sparam in
      let u =
        { t_mod  = PS.singleton var ;
          t_dep  = PM.add var (PS.singleton param) u.t_dep ;
          t_diff = PM.add var !ref_all_ps u.t_diff ;
          t_v    = PS.empty } in
      let r = T_ok u in
      sanity_check "param" r;
      if dbg_param then
        F.printf "Comp,param (%s,%s):\n%a" x sparam (fp "  ") r;
      r
    let pyro_module (x: string) ((param, isdiff): string * bool): t =
      let u = !ref_uid in
      let var = PVar x in
      let diff =
        if isdiff then !ref_all_ps else PS.remove (PParDens param) !ref_all_ps in
      let u =
        { t_mod  = PS.singleton var ;
          t_dep  = PM.add var (PS.singleton (PParDens param)) u.t_dep ;
          t_diff = PM.add var diff u.t_diff ;
          t_v    = PS.empty } in
      let r = T_ok u in
      sanity_check "module" r;
      if dbg_module then
        F.printf "Comp,module (%s,%s):\n%a" x param (fp "  ") r;
      r
    let ftop _ = false
    let pyro_sample
        ?(safei: (int -> bool) * (int -> bool) * bool = (ftop,ftop,false))
        ~(repar: bool)
        (x: string) (param: string)
        (dist_diff, dist_sig) ((n, el): expr * expr list): t =
      let okdiv, okpar, okexp = safei in
      let u = !ref_uid in
      let var = PVar x in
      let param = PParDens param in
      let del = List.map dep_expr_ps el in
      let alldep = List.fold_left PS.union PS.empty del in
      let alldep_par = PS.add param alldep in
      let dep =
        PM.add var (PS.singleton param)
          (PM.add PDens (PS.add PDens alldep_par) u.t_dep) in
      let diff =
        let diffcall =
          let diff_res =
            call_diff_info ~safei:okdiv (dist_sig, del, el) in
          if dist_diff then diff_res
          else PS.remove param diff_res in
        (* Adding discontinuities for parameters that may not be ok *)
        let discont, _ =
          List.fold_left
            (fun (acc, i) ex ->
              let acc =
                if okpar i then acc
                else PS.union acc (dep_expr_ps ex) in
              acc, i+1
            ) (PS.empty, 0) el in
        let discont =
          if okexp then discont
          else PS.union discont (dep_expr_ps n) in
        if dbg_sample then
          F.printf "Comp,Sample (%b,%d,%a): discont=%a\n   dep:  %a\n   ndiff:%a\n"
            dist_diff (List.length el) par_fp param ps_fp discont
            ps_fp alldep_par ps_fp (PS.diff !ref_all_ps diffcall);
        let diffcall = PS.diff diffcall discont in
        PM.add var !ref_all_ps
          (PM.add PDens diffcall u.t_diff) in
      let r = T_ok { t_mod  = PS.add param (PS.singleton var) ;
                     t_dep  = dep ;
                     t_diff = diff ;
                     t_v    = PS.empty } in
      if dbg_sample then
        F.printf "Comp,sample result:\n%a\n" (fp "  ") r;
      sanity_check "sample output" r;
      r
    let pyro_observe
        ?(safei: (int -> bool) * (int -> bool) * bool = (ftop,ftop,false))
        (dist_diff, dist_sig) (el: expr list) (e: expr): t =
      let okdiv, okpar, okexp = safei in
      let u = !ref_uid in
      let del = List.map dep_expr_ps el in
      let alldep =
        List.fold_left PS.union (dep_expr_ps e) del in
      let dep =
        PM.add PDens (PS.add PDens alldep) u.t_dep in
      let diff =
        let diffcall, diff_res =
          let diff_res =
            call_diff_info ~safei:okdiv (dist_sig, del, el) in
          if dist_diff then
            PS.inter (diff_expr ~safe:okexp e) diff_res, diff_res
          else
            PS.diff diff_res (dep_expr_ps e), diff_res in
        if dbg_observe then
          F.printf "Comp,Observe,ndiffcall:\nndiffcall: %a\nndiffres: %a\ndep: %a\n"
            ps_fp (PS.diff !ref_all_ps diffcall)
            ps_fp (PS.diff !ref_all_ps diff_res) ps_fp (dep_expr_ps e);
        (* Adding discontinuities for parameters that may not be ok *)
        let discont, _ =
          List.fold_left
            (fun (acc, i) ex ->
              if false  && not (okpar i) then
                F.printf "discont parameter %d: %a\n" i ps_fp (dep_expr_ps ex);
              let acc =
                if okpar i then acc
                else PS.union acc (dep_expr_ps ex) in
              acc, i+1
            ) (PS.empty, 0) el in
        let diffcall = PS.diff diffcall discont in
        if dbg_observe then
          F.printf "Comp,Observe (%b,%d): %a\n   dep:  %a\n   ndiff:%a\n"
            dist_diff (List.length el) ps_fp discont
            ps_fp alldep ps_fp (PS.diff !ref_all_ps diffcall);
        PM.add PDens diffcall u.t_diff in
      let r = T_ok { t_mod  = PS.singleton PDens ;
                     t_dep  = dep ;
                     t_diff = diff ;
                     t_v    = PS.empty } in
      if dbg_observe then
        F.printf "Comp,observe result:\n%a\n" (fp "  ") r;
      sanity_check "observe output" r;
      r
    let loop_condition (depvars: SS.t) (t: t): t =
      let r =
        match t with
        | T_ok u ->
            let dpars = SS.fold (fun v -> PS.add (PVar v)) depvars PS.empty in
            T_ok { u with t_v = dpars }
        | T_err _ -> t in
      sanity_check "loop_condition" r;
      r
    let condition (depvars: SS.t) (t: t): t =
      let r =
        match t with
        | T_ok u ->
            let pdep = vars_to_ps depvars in
            let fm op m =
              PS.fold
                (fun p acc ->
                  let s = try PM.find p acc with Not_found -> PS.empty in
                  let s = op s pdep in
                  PM.add p s acc
                ) u.t_mod m in
            T_ok { u with
                   t_dep  = fm PS.union u.t_dep ;
                   t_diff = fm PS.diff u.t_diff }
        | T_err _ -> t in
      sanity_check "condition" r;
      r
    (* Get differentiability information for density (diff,non diff) *)
    let get_density_diff_info (t: t): diff_info =
      match t with
      | T_ok u ->
          let f m =
            PS.fold
              (function
                | PVar _ | PDens -> fun a -> a
                | PParDens s -> SS.add s
              ) m SS.empty in
          let diff = try PM.find PDens u.t_diff with Not_found -> PS.empty in
          let pdiff = f diff in
          { di_dens_diff  = pdiff ;
            di_dens_ndiff = SS.diff !ref_all_pars pdiff ;
            di_prb_ndiff  = SM.empty ;
            di_val_ndiff  = SM.empty }
      | T_err _ -> failwith "error: no differentiability information"
  end: DOM_DIFF_COMP)

(** Full compositional abstraction:
 **  with a representation that tracks variables/parameters with respect to which
 **  differentiability MAY NOT hold *)
module CD_ndiff =
  (struct
    let r_goal = ref Top
    let name () = F.asprintf "compositional %a" fp_diff_prop !r_goal
    let isnontop = true
    (* Some utilities, put here for now, maybe move later *)
    type par =
      | PVar of string     (* Program variable *)
      | PParRdb of string  (* Random parameter value in initial RDB: mu     *)
      | PParPrb of string  (* Random parameter probability density:  pr_mu  *)
      | PParVal of string  (* Random parameter value after sampling: val_mu *)
      | PDens              (* Density of execution (like) *)
    module POrd =
      struct
        type t = par
        let compare = compare
      end
    module PS = Set.Make( POrd )
    module PM = Map.Make( POrd )
    let pm_updates l a = List.fold_left (fun a (x,y) -> PM.add x y a) a l
    (* Abstractions of transformations *)
    type u =
        { (* modified parameters
           *    set of parameters that may be modified by a command *)
          t_mod:  PS.t ;
          (* dependency partial map
           *    when x is not in the map, it means x=>{x}
           *    which means that x depends only on itself *)
          t_dep:  PS.t PM.t;
          (* non-differentiability partial map
           *    when x is not in the map, it means c is fifferentiable
           *    wrt all variables *)
          t_ndiff: PS.t PM.t;
          (* possible (non)-termination dependencies
           *    variables on which the termination of the command may depend *)
          t_v: PS.t }
    type t =
      | T_ok of u (* to fill *)
      | T_err of string
    (* Set of parameters *)
    let ref_all_pars: SS.t ref = ref SS.empty
    let ref_all_vars: SS.t ref = ref SS.empty
    let ref_all_ps:   PS.t ref = ref PS.empty
    let ref_uid: u ref = ref { t_mod   = PS.empty ;
                               t_dep   = PM.empty ;
                               t_ndiff = PM.empty ;
                               t_v     = PS.empty }
    let ref_id: t ref =
      (* before initialisation *)
      ref (T_err "undefined init")
    let init_domain ~(goal: diff_prop) ~(params: SS.t) ~(vars: SS.t): unit =
      r_goal := goal;
      if dbg_init then
        F.printf "Domain init:\n - parameters: %a\n - vars: %a\n"
          ss_fp params ss_fp vars;
      ref_all_pars := params;
      ref_all_vars := vars;
      ref_all_ps :=
        begin
          let gen f = SS.fold (fun x -> PS.add (f x)) !ref_all_pars in
          let s = SS.fold (fun x -> PS.add (PVar x)) !ref_all_vars (PS.singleton PDens) in
          let s = gen (fun x -> PParRdb x) s in
          let s = gen (fun x -> PParPrb x) s in
          let s = gen (fun x -> PParVal x) s in
          s
        end;
      let uid =
        let dep =
          PS.fold (fun x -> PM.add x (PS.singleton x)) !ref_all_ps PM.empty in
        let ndiff =
          PS.fold (fun x -> PM.add x PS.empty) !ref_all_ps PM.empty in
        { t_mod   = PS.empty;
          t_dep   = dep;
          t_ndiff = ndiff ;
          t_v     = PS.empty } in
      ref_uid := uid;
      ref_id := T_ok uid
    (* Temporary *)
    let error (msg: string): t = T_err msg
    (* Prtty-printing *)
    let par_fp fmt = function
      | PVar v -> F.fprintf fmt "%s" v
      | PParRdb v -> F.fprintf fmt "[%s:rdb]" v
      | PParPrb v -> F.fprintf fmt "[%s:prd]" v
      | PParVal v -> F.fprintf fmt "[%s:val]" v
      | PDens  -> F.fprintf fmt "<dens>"
    let ps_fp fmt s =
      F.fprintf fmt "{ ";
      PS.iter (F.fprintf fmt "%a; " par_fp) s;
      F.fprintf fmt "}"
    let fp (ind: string) (fmt: form) (t: t): unit =
      let subind = "    "^ind in
      let fs fmt s = ps_fp fmt s in
      match t with
      | T_ok u ->
          F.fprintf fmt "%sOk state\n%s  Mods: %a\n" ind ind fs u.t_mod;
          F.fprintf fmt "  %sDeps: (only non x -> {x} cases)\n" ind;
          PM.iter
            (fun x a ->
              if not (PS.equal a (PS.singleton x)) then
                F.fprintf fmt "%s%a => %a\n" subind par_fp x fs a
            ) u.t_dep;
          F.fprintf fmt "%s  May non-%a: (only non empty cases)\n" ind
            fp_diff_prop_short !r_goal;
          PM.iter
            (fun x a ->
              if a != PS.empty then
                F.fprintf fmt "%s%a => %a\n" subind par_fp x fs a
            ) u.t_ndiff;
          F.fprintf fmt "%s  V: %a\n" ind fs u.t_v
      | T_err s -> F.fprintf fmt "%sKo state [ %s ]\n" ind s
    (* Some functions for debugging *)
    let crash msg =
      flush stdout;
      failwith msg
    let sanity_check (msg: string) (t: t): unit =
      let errors = ref [ ] in
      (* check well-formedness of a map to sets:
       *  - keys should be exactly the set of all dimensions
       *  - each set should be included in the set of dimensions *)
      let check (name: string) (m: PS.t PM.t): unit =
        let add_error msg =
          errors := (name, msg) :: !errors in
        PS.iter
          (fun i ->
            if not (PM.mem i m) then
              add_error (F.asprintf "missing par %a" par_fp i)
          ) !ref_all_ps;
        PM.iter
          (fun i s ->
            if not (PS.mem i !ref_all_ps) then
              add_error (F.asprintf "unbound map par %a" par_fp i);
            PS.iter
              (fun i ->
                if not (PS.mem i !ref_all_ps) then
                  add_error (F.asprintf "unbound set par %a" par_fp i)
              ) s
          ) m in
      let do_u (u: u): unit =
        (* modified variables should all be defined *)
        if not (PS.subset u.t_mod !ref_all_ps) then
          PS.iter
            (fun p ->
              let msg = F.asprintf "%a found" par_fp p in
              errors := ("modified pars", msg) :: !errors
            ) (PS.diff !ref_all_ps u.t_mod);
        (* maps of dependences and differentiability should be well-formed *)
        check "dep" u.t_dep;
        check "diff" u.t_ndiff;
        (* for each variable, non differentiable pars should also appear in
         * dependences *)
        PM.iter
          (fun i d ->
            let np = try PM.find i u.t_ndiff with Not_found -> PS.empty in
            if not (PS.subset np d) then
              errors := ("!(ndiff<=dep)", F.asprintf "%a" par_fp i) :: !errors
          ) u.t_dep;
        (* error report *)
        if !errors != [] then
          begin
            F.printf "Validity condition failed (%s):\n  All: %a\n" msg ps_fp
              !ref_all_ps;
            List.iter
              (fun (name, error) -> F.printf "  in %s: %s\n" name error)
              (List.rev !errors);
            F.printf "In abstract state:\n%a\n" (fp "  ") t
          end in
      if do_sanity_checks then
        match t with
        | T_ok u -> do_u u
        | T_err _ -> ()
    (* Abstraction of identity function *)
    let id (): t =
      let r = !ref_id in
      sanity_check "id" r;
      r
    (* Composition *)
    let compose (t0: t) (t1: t): t =
      sanity_check "compose-l" t0;
      sanity_check "compose-r" t1;
      (* TODO: issue: we need to require allpars here,
       *              which is not satisfactory *)
      match t0, t1 with
      | T_ok u0, T_ok u1 ->
          let get_dep0 x =
            try PM.find x u0.t_dep with Not_found -> PS.empty in
          let get_deps0 s =
            PS.fold (fun y -> PS.union (get_dep0 y)) s PS.empty in
          let get_ndiff0 x =
            try PM.find x u0.t_ndiff
            with Not_found -> crash "" in
          let get_ndiffs0 s =
            PS.fold (fun y -> PS.union (get_ndiff0 y)) s PS.empty in
          let dep =
            PM.fold
              (fun x d1 acc ->
                let d = PS.fold (fun y -> PS.union (get_dep0 y)) d1 u0.t_v in
                PM.add x d acc
              ) u1.t_dep PM.empty in
          let ndiff =
            PM.fold
              (fun x np1 acc ->
                let d1 = try PM.find x u1.t_dep with Not_found -> PS.empty in
                if false && not (PS.subset np1 d1) then
                  F.printf "WARN: inclusion: %a\n" ps_fp (PS.diff np1 d1);
                let p1 = PS.diff !ref_all_ps np1 in
                let np =
                  PS.union (get_deps0 np1)
                    (get_ndiffs0 (PS.inter d1 p1)) in
                PM.add x (PS.union u0.t_v np) acc
              ) u1.t_ndiff PM.empty in
          let v =
            PS.fold
              (fun x acc ->
                try PS.union (PM.find x u0.t_dep) acc
                with Not_found -> acc
              ) u1.t_v u0.t_v in
          let r = T_ok { t_mod   = PS.union u0.t_mod u1.t_mod ;
                         t_dep   = dep ;
                         t_ndiff = ndiff ;
                         t_v     = v } in
          sanity_check "compose-out" r;
          if dbg_compose then
            F.printf "Compose:\n%a\n%a\n%a" (fp "     ") t0
              (fp "     ") t1 (fp "   ") r;
          r
      | T_ok _, _ -> t1
      | _, _ -> t0
    (* Lattice operations *)
    let join (t0: t) (t1: t): t =
      let t =
        match t0, t1 with
        | T_ok u0, T_ok u1 ->
            let fm m0 m1 =
              PM.fold
                (fun p s0 acc ->
                  let s1 =
                    try PM.find p m1
                    with Not_found -> failwith "incorrect state" in
                  PM.add p (PS.union s0 s1) acc
                ) m0 PM.empty in
            T_ok { t_mod   = PS.union u0.t_mod u1.t_mod ;
                   t_dep   = fm u0.t_dep u1.t_dep ;
                   t_ndiff = fm u0.t_ndiff u1.t_ndiff;
                   t_v     = PS.union u0.t_v u1.t_v }
        | T_ok _, _ -> t1
        | _, _ -> t0 in
      if dbg_join then
        F.printf "Join:\n- L:\n%a- R:\n%a===>\n%a\n"
          (fp "  ") t0 (fp "  ") t1 (fp "    ") t;
      t
    let equal (t0: t) (t1: t): bool =
      try
        match t0, t1 with
        | T_ok u0, T_ok u1 ->
            if not (PS.equal u0.t_mod u1.t_mod) then raise Stop;
            let f m0 m1 =
              let g m = PM.fold (fun p _ -> PS.add p) m PS.empty in
              if not (PS.equal (g m0) (g m1)) then raise Stop;
              PM.iter
                (fun p s0 ->
                  let s1 = try PM.find p m1 with Not_found -> raise Stop in
                  if not (PS.equal s0 s1) then raise Stop) m0 in
            f u0.t_dep u1.t_dep;
            f u0.t_ndiff u1.t_ndiff;
            if not (PS.equal u0.t_v u1.t_v) then raise Stop;
            true
        | T_err _, T_err _ -> true
        | _, _ -> false
      with Stop -> false
    (* Abstraction of basic operations *)
    let vars_to_ps (s: SS.t): PS.t =
      SS.fold (fun x -> PS.add (PVar x)) s PS.empty
    let dep_expr_ps (e: expr) = vars_to_ps (dep_expr e)
    let rec ndiff_expr ?(safe: bool = false) (e: expr): PS.t =
      let rec aux_expr (acc: PS.t): expr -> PS.t = function
        | Nil | True | False | Num _ | Str _ | Name _ | Ellipsis -> acc
        | BOp ((Add | Sub | Mult | Pow), e0, e1) ->
            aux_expr (aux_expr acc e0) e1
        | BOp (Div, e0, e1) ->
            if safe then aux_expr (aux_expr acc e0) e1
            else PS.union (aux_expr (aux_expr acc e0) e1) (dep_expr_ps e1)
        | Comp (_, e0, e1) ->
            PS.union (dep_expr_ps e0) (dep_expr_ps e1)
        | List el ->
            List.fold_left (fun acc e -> aux_expr acc e) acc el
        | _ ->
            F.printf "TODO[diff_expr]:\n  %a\n\n\n" fp_expr e; flush stdout;
            failwith "todo: diff_expr" in
      aux_expr PS.empty e
    let assign ?(safe: bool = false) (x: idtf) (e: expr): t =
      let u = !ref_uid in
      let pvar = PVar x in
      let r =
        T_ok { t_mod   = PS.singleton pvar ;
               t_dep   = PM.add pvar (dep_expr_ps e) u.t_dep ;
               t_ndiff = PM.add pvar (ndiff_expr ~safe e) u.t_ndiff ;
               t_v     = PS.empty } in
      sanity_check "assign" r;
      r
    let call_ndiff_info ?(safei: int -> bool = fun _ -> false) c =
      let rec aux i ndiff = function
        | b :: fsig, de :: del, e :: eas ->
            (*let old = diff in*)
            let ndiff =
              if b then (* parameter is differentiable *)
                PS.union (ndiff_expr ~safe:(safei i) e) ndiff
              else (* parameter not differentiable *)
                PS.union ndiff de in
            aux (i+1) ndiff (fsig, del, eas)
        | [ ], [ ], [ ] -> ndiff
        | [ ], de :: del, _ :: eas ->
            aux (i+1) (PS.union ndiff de) ([ ], del, eas)
        | _, _, _ -> failwith (F.asprintf "incoherent call: %s" (name ())) in
      aux 0 PS.empty c
    let call
        (name: string)
        ((is_ipdiff,fsig): bool * bool list)
        (xo: idtf option) ((ef, eas): expr * expr list): t =
      let r =
        match xo with
        | None ->
            error "call proc, todo"
        | Some x ->
            let dep_f =
              if dbg_call && not is_ipdiff then
                F.printf "nonsmooth,dep_f : %b\n" (SS.mem name !ref_all_pars);
              PS.singleton (PParRdb name) in
            if dbg_call && not is_ipdiff then
              F.printf "nonsmooth,dep_f : %a\n" ps_fp dep_f;
            let del = List.map dep_expr_ps eas in
            let dep = List.fold_left PS.union dep_f del in
            let ndiff =
              let ndiff = call_ndiff_info (fsig, del, eas) in
              if is_ipdiff then ndiff
              else PS.union ndiff dep_f in
            let u = !ref_uid in
            let var = PVar x in
            let u = { t_mod   = PS.singleton var ;
                      t_dep   = PM.add var dep u.t_dep ;
                      t_ndiff = PM.add var ndiff u.t_ndiff ;
                      t_v     = PS.empty } in
            T_ok u in
      sanity_check "call" r;
      if dbg_call then
        F.printf "Comp,call: %s\n%a\n" name (fp "  ") r;
      r
    let pyro_param (x: string) (sparam: string): t =
      (* interpret as x = theta *)
      let u = !ref_uid in
      let var = PVar x
      and param = PParRdb sparam (* theta *) in
      let u =
        { t_mod   = PS.singleton var ;
          t_dep   = PM.add var (PS.singleton param) u.t_dep ;
          t_ndiff = PM.add var PS.empty u.t_ndiff ;
          t_v     = PS.empty } in
      let r = T_ok u in
      sanity_check "param" r;
      if dbg_param then
        F.printf "Comp,param (%s,%s):\n%a" x sparam (fp "  ") r;
      r
    let pyro_module (x: string) ((sparam, isdiff): string * bool): t =
      (* interpret as x(y) = f(y,theta) *)
      let u = !ref_uid in
      let var = PVar x
      and param = PParRdb sparam (* theta *) in
      let ndiff =
        if isdiff then PS.empty else PS.singleton param in
      let u =
        { t_mod   = PS.singleton var ;
          t_dep   = PM.add var (PS.singleton param) u.t_dep ;
          t_ndiff = PM.add var ndiff u.t_ndiff ;
          t_v     = PS.empty } in
      let r = T_ok u in
      sanity_check "module" r;
      if dbg_module then
        F.printf "Comp,module (x:%s, param:%s, %b):\n%a" x sparam isdiff (fp "  ") r;
      r
    let ftop _ = false
    let pyro_sample
        ?(safei: (int -> bool) * (int -> bool) * bool = (ftop,ftop,false))
        ~(repar: bool)
        (x: string) (param: string)
        (dist_diff, dist_sig) ((namex, el): expr * expr list): t =
      let okdiv, okpar, okexp = safei in
      let u = !ref_uid in
      let var_x  = PVar x in        (* x *)
      let par_mu = PParRdb param in (* mu *)
      let par_v  = PParVal param in (* val_mu *)
      let par_d  = PParPrb param in (* pr_mu *)
      let del    = List.map dep_expr_ps el in
      let dex    = dep_expr_ps namex in
      if dbg_sample then
        begin
          F.printf "Comp,Sample,info: distdiff: %b (%d), paramd: %a, repar: %b\n"
            dist_diff (List.length el) par_fp par_d repar;
          F.printf " safety: %b\n" okexp
        end;
      (* Dependencies *)
      let dep =
        let dep_del = List.fold_left PS.union PS.empty del in
        let dep_var_x, dep_par_v, dep_par_d =
          if repar then (* repar case *)
            let dep_var_x = PS.add par_mu (PS.union dep_del dex) in
            let dep_par_v = dep_var_x in
            let dep_par_d = PS.singleton par_mu in
            dep_var_x, dep_par_v, dep_par_d
          else          (* non repar case *)
            let dep_var_x = PS.singleton par_mu in
            let dep_par_v = dep_var_x in
            let dep_par_d = PS.add par_mu (PS.union dep_del dex) in
            dep_var_x, dep_par_v, dep_par_d in
        if dbg_sample then
          F.printf "Comp,Sample,deps: %a\n" ps_fp dep_del;
        pm_updates [ var_x, dep_var_x ;
                     par_v, dep_par_v ;
                     par_d, dep_par_d ] u.t_dep in
      (* Differentiability *)
      let ndiff =
        let ndiffcall =
          let ndiff_res =
            call_ndiff_info ~safei:okdiv (dist_sig, del, el) in
          if dist_diff then ndiff_res
          else PS.add par_mu ndiff_res in
        (* - adding discontinuities for parameters that may not be error free *)
        let discont, _ =
          if repar then
            PS.empty, 0
          else
            List.fold_left
              (fun (acc, i) ex ->
                if dbg_sample then
                  F.printf " arg %d: %s\n" i (if okpar i then "ok" else "KO");
                let acc = if okpar i then acc else PS.union acc (dep_expr_ps ex) in
                acc, i+1
              ) (PS.empty, 0) el in
        let discont = if okexp then discont else PS.union discont dex in
        if dbg_sample then
          F.printf "Comp,Sample,diff:\n   discont=%a\n   ndiff:%a\n"
            ps_fp discont ps_fp ndiffcall;
        let ndiff_var_x, ndiff_par_v, ndiff_par_d =
          let ndiffcall = PS.union ndiffcall discont in
          if repar then
            let ndiff_var_x = PS.union dex ndiffcall in
            let ndiff_par_v = PS.union dex ndiffcall in
            let ndiff_par_d = if dist_diff then PS.empty else PS.singleton par_mu in
            ndiff_var_x, ndiff_par_v, ndiff_par_d
          else
            let ndiff_var_x = dex in
            let ndiff_par_v = dex in
            let ndiff_par_d = (*PS.union dex*) ndiffcall in
            ndiff_var_x, ndiff_par_v, ndiff_par_d in
        pm_updates [ var_x, ndiff_var_x ;
                     par_v, ndiff_par_v ;
                     par_d, ndiff_par_d ] u.t_ndiff in
      let r = T_ok { t_mod   = PS.of_list [ par_d ; par_v ; var_x ] ;
                     t_dep   = dep ;
                     t_ndiff = ndiff ;
                     t_v     = PS.empty } in
      sanity_check "sample output" r;
      if dbg_sample then
        F.printf "Comp,Sample,result:\n%a\n\n" (fp "  ") r;
      r
    let pyro_observe
        ?(safei: (int -> bool) * (int -> bool) * bool = (ftop,ftop,false))
        (dist_diff, dist_sig) (el: expr list) (e: expr): t =
      let okdiv, okpar, okexp = safei in
      let u = !ref_uid in
      if dbg_observe then
        F.printf "Comp,Observe,info: distdiff: %b (%d)\n"
          dist_diff (List.length el);
      let del = List.map dep_expr_ps el in
      (* Dependencies *)
      let dep =
        let dep_del = List.fold_left PS.union (dep_expr_ps e) del in
        if dbg_observe then
          F.printf "Comp,Observe,deps: %a\n" ps_fp dep_del;
        PM.add PDens (PS.add PDens dep_del) u.t_dep in
      (* Differentiability *)
      let ndiff =
        let ndiffcall, ndiff_res =
          let ndiff_res =
            call_ndiff_info ~safei:okdiv (dist_sig, del, el) in
          if dist_diff then
            PS.union (ndiff_expr ~safe:okexp e) ndiff_res, ndiff_res
          else
            PS.union ndiff_res (dep_expr_ps e), ndiff_res in
        (* Adding discontinuities for parameters that may not be ok *)
        let discont, _ =
          List.fold_left
            (fun (acc, i) ex ->
              if false && not (okpar i) then
                F.printf "discont parameter %d: %a\n" i ps_fp (dep_expr_ps ex);
              let acc =
                if okpar i then acc
                else PS.union acc (dep_expr_ps ex) in
              acc, i+1
            ) (PS.empty, 0) el in
        if dbg_observe then
          F.printf
            "Comp,Observe,diff:\n ndiffcall: %a\n ndiffres: %a\n discont: %a\n"
            ps_fp ndiffcall ps_fp ndiff_res ps_fp discont;
        let ndiffcall = PS.union ndiffcall discont in
        PM.add PDens ndiffcall u.t_ndiff in
      let r = T_ok { t_mod   = PS.singleton PDens ;
                     t_dep   = dep ;
                     t_ndiff = ndiff ;
                     t_v     = PS.empty } in
      sanity_check "observe output" r;
      if dbg_observe then
        F.printf "Comp,Observe,result:\n%a\n" (fp "  ") r;
      r
    let loop_condition (depvars: SS.t) (t: t): t =
      let r =
        match t with
        | T_ok u ->
            let dpars = SS.fold (fun v -> PS.add (PVar v)) depvars PS.empty in
            T_ok { u with t_v = PS.union u.t_v dpars }
        | T_err _ -> t in
      sanity_check "loop_condition" r;
      r
    let condition (depvars: SS.t) (t: t): t =
      let r =
        match t with
        | T_ok u ->
            let pdep = vars_to_ps depvars in
            let fm op m =
              PS.fold
                (fun p acc ->
                  let s = try PM.find p acc with Not_found -> PS.empty in
                  let s = op s pdep in
                  PM.add p s acc
                ) u.t_mod m in
            T_ok { u with
                   t_dep   = fm PS.union u.t_dep ;
                   t_ndiff = fm PS.union u.t_ndiff }
        | T_err _ -> t in
      sanity_check "condition" r;
      r
    (* Get differentiability information for density (diff,non diff) *)
    let get_density_diff_info (t: t): diff_info =
      match t with
      | T_ok u ->
          let f m =
            PS.fold
              (function
                | PParRdb s -> SS.add s
                | PVar _ | PDens
                | PParPrb _ | PParVal _ -> fun a -> a
              ) m SS.empty in
          let ndiff = try PM.find PDens u.t_ndiff with Not_found -> PS.empty in
          let pndiff = f ndiff in
          let prb_ndiff, val_ndiff =
            PM.fold
              (fun p s (acc_prb, acc_val) ->
                match p with
                | PParPrb n -> SM.add n (f s) acc_prb, acc_val
                | PParVal n -> acc_prb, SM.add n (f s) acc_val
                | _ -> acc_prb, acc_val
              ) u.t_ndiff (SM.empty, SM.empty) in
          { di_dens_diff  = SS.diff !ref_all_pars pndiff ;
            di_dens_ndiff = pndiff ;
            di_prb_ndiff  = prb_ndiff ;
            di_val_ndiff  = val_ndiff }
      | T_err msg ->
          failwith (F.asprintf "no diff info (%s)" msg)
  end: DOM_DIFF_COMP)
