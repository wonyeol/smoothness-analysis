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
 ** batch_diff.ml: entry point for batch execution of diff analysis *)
open Analysis_sig
open Ai_diff
open Lib

open Ddom_sig

open Diff_util

(** Type for regression test data *)
(* type for regression test datum *)
type rt_entry =
    { (* regression test name *)
      rte_name:        string ;
      (* file to analyze *)
      rte_filename:    string ;
      (* funcs to inline *)
      rte_inlinefns: string list ;
      (* !!! TODO: add optional overriding abstract domain info *)
      (* WL: unused.
      (* expected result:
       *   None:   no regression information available;
       *   Some l: we expect differentiability wrt the pars in l *)
      rte_ddiffvars:   string list option ;
       *)
      (* expected result:
       *   None: no regression information available;
       *   Some [(s_1, dp_1);...]: we expect that for each dp : diff_prop,
       *     the density is dp w.r.t. {s_i : dp_i <= dp}. *)
      rte_diffinfo: (string * diff_prop) list option;
      (* all random variables *)
      rte_rvars: string list;
      (* all discrete random variables *)
      rte_rvars_disc: string list;
      (* guide parameters:
       *  - case of model: empty
       *  - case of guide: list of optimisation parameters
       *)
      rte_guide_pars: string list }

(* type for regression test data *)
type rt_table = rt_entry list

(** Regression test data *)
let tests_diff_all: rt_table =
  [
   (* Complexity of examples:
    *   air                    (has loops and user-defined funcs)
    *   >> dmm                 (has loops but no user-defined funcs)
    *   >> lda ~ sgdef ~ ssvae (has no loops and no user-defined funcs)
    *   > br ~ csis ~ vae      (shorter LoC) *)

   (* Examples for POPL'20. *)
   (* air *)
   (
    let params = [
      ("decode_l1", Lips); (* applied to F.relu(...). *)
      ("decode_l2", Diff);
    ] in
    let rvars = [
      (* ("data",       Top);  (* subsample distribution. *) *)
      ("z_pres_{}",  Top);  (* discrete distribution. *)
      ("z_where_{}", Lips); (* applied to F.grid_sample(_, ...). *)
      ("z_what_{}",  Lips); (* applied to F.relu(...). *)
      (* Reason for Lips:
       * - "z_where_{}":
       *     cur_z_where = pyro.sample(..."z_where_{}"...)
       *     out = ...cur_z_where...
       *     ...out...
       *     theta = out
       *     grid = ...theta...
       *     ... = F.grid_sample(..., grid)
             * - "z_what_{}":
       *     cur_z_what = pyro.sample(..."z_what_{}"...)
       *     ... = ...F.relu(...cur_z_what...)... *)
      ] in
    let rvars_disc = [ "z_pres_{}" ] in
    { rte_name       = "air (1-model)";
      rte_filename   = "../srepar/srepar/examples/air/model.py";
      rte_inlinefns  = ["main"];
      rte_diffinfo   = Some (params @ rvars);
      rte_rvars      = List.map fst rvars;
      rte_rvars_disc = rvars_disc;
      rte_guide_pars = List.map fst params }
   );

   (
    let params = [
      ("rnn",           Lips); (* applied to F.relu(...). *)
      ("bl_rnn",        Lips); (* applied to F.relu(...). *)
      ("predict_l1",    Lips); (* applied to F.relu(...). *)
      ("predict_l2",    Diff);
      ("encode_l1",     Lips); (* applied to F.relu(...). *)
      ("encode_l2",     Diff);
      ("bl_predict_l1", Lips); (* applied to F.relu(...). *)
      ("bl_predict_l2", Diff);
      ("h_init",        Lips); (* applied to F.relu(...). *)
      ("c_init",        Lips); (* applied to F.relu(...). *)
      ("z_where_init",  Lips); (* applied to F.relu(...). *)
      ("z_what_init",   Lips); (* applied to F.relu(...). *)
      ("bl_h_init",     Lips); (* applied to F.relu(...). *)
      ("bl_c_init",     Lips); (* applied to F.relu(...). *)
      (* Reason for Lips:
       * - "rnn", "h_init", "c_init", "z_where_init", "z_what_init":
       *     state_h = ...h_init...
       *     state_c = ...c_init...
       *     state_z_where = ...z_where_init...
       *     state_z_what = ...z_what_init...
       *     rnn_input = ...state_z_where...state_z_what...
       *     state_h, ... = ...rnn...state_h...state_c...rnn_input...
       *     ... = ...F.relu(...state_h...)...
       * - "bl_rnn", "bl_h_init", "bl_c_init":
       *     state_bl_h = ...bl_h_init...
       *     state_bl_c = ...bl_c_init...
       *     state_bl_h, ... = ...bl_rnn...state_bl_h...state_bl_c...
       *     ... = ...F.relu(...state_bl_h...)... *)
      (* NOTE:
       * - The parameters "bl_..." are all used to compute `baseline_value`
       *   of the sample statement for "z_pres_{}". In the original code,
       *   the value of `baseline_value` is passed to the kwarg `infer`
       *   of `pyro.sample` for "z_pres_{}", but the kwargs is ignored by
       *   our analyser on the ground that it does not affect densities.
       * - Though not affecting densities, `baseline_value` does affect
       *   the value of gradient estimate of ELBO (e.g., in SCORE estimator),
       *   in a differentiable way. In more detail, it affects the value of
       *   gradient estimate, not through densities, but through something
       *   related to control variate. And we need to make our analyser
       *   aware of this fact to ensure its soundness.
       * - Since the way that `baseline_value` is passed to `pyro.sample`
       *   involves `dict` objects in Python, our analyser does not analyse
       *   the original guide directly. Instead, the following lines are
       *   inserted to the original guide to make an equivalent effect
       *   for "bl_..." describe above:
       *     p = torch.exp(bl_value - bl_value)
       *     pyro.sample(_, Normal(p,1), obs=0)
       *   where bl_value is a computed value of `baseline_value`. In this way,
       *   "bl_..." now affects the densities directly, but the above observe
       *   statement makes a constant score, thereby not changing the behavior
       *   of the original guide. *)
    ] in
    let rvars = [
      (* ("data",       Top);  (* subsample distribution. *) *)
      ("z_pres_{}",  Top);  (* discrete distribution. *)
      ("z_where_{}", Top);  (* applied to F.relu(...) and _/(...). *)
      ("z_what_{}",  Lips); (* applied to F.relu(...). *)
      (* Reason for Lips:
       * - "z_where_{}", "z_what_{}":
       *     cur_z_where = pyro.sample(..."z_where_{}"...)
       *     cur_z_what = pyro.sample(..."z_what_{}"...)
       *     state_z_where = cur_z_where
       *     state_z_what = cur_z_what
       *     [next iteration of a loop]
       *     rnn_input = ...state_z_where...state_z_what...
       *     state_h, ... = ...rnn_input...
       *     ... = ...F.relu(...state_h...)... *)
      (* NOTE:
       * - diff_prop of "z_where_{}" is Top, not Lips,
       *   since it is used as a denomiator of a division:
       *     cur_z_where = pyro.sample("z_where_{}"..., Normal...)
       *     out = ...cur_z_where...
       *     out = out / (...cur_z_where...)
       * - [Q] does this produce any NaN? If so, pointing this out would be helpful. *)
    ] in
    let rvars_disc = [ "z_pres_{}" ] in
    { rte_name       = "air (2-guide)";
      rte_filename   = "../srepar/srepar/examples/air/guide.py";
      rte_inlinefns  = ["main"];
      rte_diffinfo   = Some (params @ rvars);
      rte_rvars      = List.map fst rvars;
      rte_rvars_disc = rvars_disc;
      rte_guide_pars = List.map fst params; }
   ) ;

   (* br *)
   (
    let params = [] in
    let rvars = [
      ("a",     Diff);
      ("bA",    Diff);
      ("bR",    Diff);
      ("bAR",   Diff);
      ("sigma", Diff);
    ] in
    let rvars_disc = [ ] in
    { rte_name       = "br (1-model)";
      rte_filename   = "../srepar/srepar/examples/br/model.py";
      rte_inlinefns  = ["main"];
      rte_diffinfo   = Some (params @ rvars);
      rte_rvars      = List.map fst rvars;
      rte_rvars_disc = rvars_disc;
      rte_guide_pars = List.map fst params}
   );
   (
    let params = [
      ("a_loc",         Diff);
      ("a_scale",       Diff);
      ("sigma_loc",     Diff);
      ("weights_loc",   Diff);
      ("weights_scale", Diff);
    ] in
    let rvars = [
      ("a",     Diff);
      ("bA",    Diff);
      ("bR",    Diff);
      ("bAR",   Diff);
      ("sigma", Diff);
    ] in
    let rvars_disc = [ ] in
    { rte_name      = "br (2-guide)";
      rte_filename  = "../srepar/srepar/examples/br/guide.py";
      rte_inlinefns = ["main"];
      rte_diffinfo  = Some (params @ rvars);
      rte_rvars      = List.map fst rvars;
      rte_rvars_disc = rvars_disc;
      rte_guide_pars = List.map fst params; };
   );

   (* csis *)
   (
    let params = [] in
    let rvars = [
      ("z", Diff);
    ] in
    let rvars_disc = [ ] in
    { rte_name       = "csis (1-model)";
      rte_filename   = "../srepar/srepar/examples/csis/model.py";
      rte_inlinefns  = ["main"];
      rte_diffinfo   = Some (params @ rvars);
      rte_rvars      = List.map fst rvars;
      rte_rvars_disc = rvars_disc;
      rte_guide_pars = List.map fst params; }
   );
   (
    let params = [
      ("first",  Lips); (* applied to nn.ReLU()(...). *)
      ("second", Lips); (* applied to nn.ReLU()(...). *)
      ("third",  Lips); (* applied to nn.ReLU()(...). *)
      ("fourth", Lips); (* applied to nn.ReLU()(...). *)
      ("fifth",  Diff);
    ] in
    let rvars = [
      ("z", Diff);
    ] in
    let rvars_disc = [ ] in
    { rte_name       = "csis (2-guide)";
      rte_filename   = "../srepar/srepar/examples/csis/guide.py";
      rte_inlinefns  = ["main"];
      rte_diffinfo   = Some (params @ rvars);
      rte_rvars      = List.map fst rvars;
      rte_rvars_disc = rvars_disc;
      rte_guide_pars = List.map fst params; };
   );

   (* dmm *)
   (
    let params = [
      ("e_lin_z_to_hidden",               Lips); (* applied to nn.ReLU()(...). *)
      ("e_lin_hidden_to_hidden",          Lips); (* applied to nn.ReLU()(...). *)
      ("e_lin_hidden_to_input",           Diff);
      ("t_lin_gate_z_to_hidden",          Lips); (* applied to nn.ReLU()(...). *)
      ("t_lin_gate_hidden_to_z",          Diff);
      ("t_lin_proposed_mean_z_to_hidden", Lips); (* applied to nn.ReLU()(...). *)
      ("t_lin_proposed_mean_hidden_to_z", Lips); (* applied to nn.ReLU()(...). *)
      ("t_lin_sig",                       Diff);
      ("t_lin_z_to_loc",                  Diff);
      (* Reason for Lips:
       * - "t_lin_proposed_mean_hidden_to_z":
       *     proposed_mean = ...(t_lin_proposed_mean_hidden_to_z)...
       *     ... = ...t_relu(proposed_mean)... *)
    ] in
    let rvars = [
      ("z_{}", Lips); (* applied to nn.ReLU()(...). *)
      (* Reason for Lips:
       * - "z_{}":
       *     z_t = pyro.sample(..."z_{}"...)
       *     ... = e_relu(...z_t...) *)
    ] in
    let rvars_disc = [ ] in
    { rte_name      = "dmm (1-model)";
      rte_filename  = "../srepar/srepar/examples/dmm/model.py";
      rte_inlinefns = ["main"];
      rte_diffinfo  = Some (params @ rvars);
      rte_rvars      = List.map fst rvars;
      rte_rvars_disc = rvars_disc;
      rte_guide_pars = List.map fst params; }
   );
   (
    let params = [
      ("c_lin_z_to_hidden",     Diff);
      ("c_lin_hidden_to_loc",   Diff);
      ("c_lin_hidden_to_scale", Diff);
      ("rnn",                   Lips); (* has `nonlinearity`="relu". *)
    ] in
    let rvars  = [
      ("z_{}", Diff);
    ] in
    let rvars_disc = [ ] in
    { rte_name       = "dmm (2-guide)";
      rte_filename   = "../srepar/srepar/examples/dmm/guide.py";
      rte_inlinefns  = ["main"];
      rte_diffinfo   = Some (params @ rvars);
      rte_rvars      = List.map fst rvars;
      rte_rvars_disc = rvars_disc;
      rte_guide_pars = List.map fst params; }
   );

   (* lda *)
   (
    let params = [] in
    let rvars  = [
      ("topic_weights", Diff);
      ("topic_words",   Diff);
      ("doc_topics",    Diff);
      ("word_topics",   Top); (* discrete distribution. *)
    ] in
    let rvars_disc = [ "word_topics" ] in
    { rte_name      = "lda (1-model)";
      rte_filename  = "../srepar/srepar/examples/lda/model.py";
      rte_inlinefns = ["main"];
      rte_diffinfo  = Some (params @ rvars);
      rte_rvars      = List.map fst rvars;
      rte_rvars_disc = rvars_disc;
      rte_guide_pars = List.map fst params; }
   );
   (
    let params = [
      ("layer1",                  Diff);
      ("layer2",                  Diff);
      ("layer3",                  Diff);
      ("topic_weights_posterior", Diff);
      ("topic_words_posterior",   Diff);
    ] in
    let rvars  = [
      ("topic_weights", Diff);
      ("topic_words",   Diff);
      (* ("documents",     Top); (* subsample distribution. *) *)
      (* ("doc_topics",    Top); *)
      (* NOTE:
       * - We do not a latent variable sampled from Dirac as a parmaeter
       *   and regard it just a named assigned variable (see ASSUMPTION 4 in
       *   `whitebox/refact/ai_diff/ai_diff.ml`).
       * - So commented out "doc_topics" in the above oracle for guide
       *   because it is such a Dirac-sampled variable. *)
    ] in
    let rvars_disc = [ ] in
    { rte_name      = "lda (2-guide)";
      rte_filename  = "../srepar/srepar/examples/lda/guide.py";
      rte_inlinefns = ["main"];
      rte_diffinfo  = Some (params @ rvars);
      rte_rvars      = List.map fst rvars;
      rte_rvars_disc = rvars_disc;
      rte_guide_pars = List.map fst params; }
   );

   (* sgdef *)
   (
    let params = [] in
    let rvars  = [
      ("w_top",    Diff);
      ("w_mid",    Diff);
      ("w_bottom", Diff);
      ("z_top",    Diff);
      ("z_mid",    Diff);
      ("z_bottom", Diff);
    ] in
    let rvars_disc = [ ] in
    { rte_name      = "sgdef (1-model)";
      rte_filename  = "../srepar/srepar/examples/sgdef/model.py";
      rte_inlinefns = ["main"];
      rte_diffinfo  = Some (params @ rvars);
      rte_rvars      = List.map fst rvars;
      rte_rvars_disc = rvars_disc;
      rte_guide_pars = List.map fst params; }
   );
   (
    let params = [
      ("log_alpha_w_q_top",    Diff);
      ("log_mean_w_q_top",     Diff);
      ("log_alpha_w_q_mid",    Diff);
      ("log_mean_w_q_mid",     Diff);
      ("log_alpha_w_q_bottom", Diff);
      ("log_mean_w_q_bottom",  Diff);
      ("log_alpha_z_q_top",    Diff);
      ("log_mean_z_q_top",     Diff);
      ("log_alpha_z_q_mid",    Diff);
      ("log_mean_z_q_mid",     Diff);
      ("log_alpha_z_q_bottom", Diff);
      ("log_mean_z_q_bottom",  Diff);
    ] in
    let rvars  = [
      ("w_top",    Diff);
      ("w_mid",    Diff);
      ("w_bottom", Diff);
      ("z_top",    Diff);
      ("z_mid",    Diff);
      ("z_bottom", Diff);
    ] in
    let rvars_disc = [ ] in
    { rte_name      = "sgdef (2-guide)";
      rte_filename  = "../srepar/srepar/examples/sgdef/guide.py";
      rte_inlinefns = ["main"];
      rte_diffinfo  = Some (params @ rvars);
      rte_rvars      = List.map fst rvars;
      rte_rvars_disc = rvars_disc;
      rte_guide_pars = List.map fst params; }
   );

   (* ssvae *)
   (
    let params = [
      ("decoder_fst", Diff);
      ("decoder_snd", Diff);
    ] in
    let rvars  = [
      ("z", Diff);
      ("y", Top); (* discrete distribution. *)
      (* NOTE:
       * - "y" is sampled if given `ys` is `None`, and observed otherwise.
       *   Since there is a trace where "y" is sampled, our analyser adds "y"
       *   to parameters. *)
    ] in
    let rvars_disc = [ "y" ] in
    { rte_name      = "ssvae (1-model)";
      rte_filename  = "../srepar/srepar/examples/ssvae/model.py";
      rte_inlinefns = ["main"];
      rte_diffinfo  = Some (params @ rvars);
      rte_rvars      = List.map fst rvars;
      rte_rvars_disc = rvars_disc;
      rte_guide_pars = List.map fst params; }
   );
   (
    let params = [
      ("encoder_y_fst",  Diff);
      ("encoder_y_snd",  Diff);
      ("encoder_z_fst",  Diff);
      ("encoder_z_out1", Diff);
      ("encoder_z_out2", Diff);
    ] in
    let rvars  = [
      ("y", Top);  (* discrete distribution. *)
      ("z", Diff);
      (* NOTE:
       * - "y" is sampled if given `ys` is `None`, and do nothing otherwise.
       *   Since there is a trace where "y" is sampled, our analyser adds "y"
       *   to parameters. *)
    ] in
    let rvars_disc = [ "y" ] in
    { rte_name      = "ssvae (2-guide)";
      rte_filename  = "../srepar/srepar/examples/ssvae/guide.py";
      rte_inlinefns = ["main"];
      rte_diffinfo  = Some (params @ rvars);
      rte_rvars      = List.map fst rvars;
      rte_rvars_disc = rvars_disc;
      rte_guide_pars = List.map fst params; }
   );

   (* vae *)
   (
    let params = [
      ("decoder_fc1",  Diff);
      ("decoder_fc21", Diff);
    ] in
    let rvars  = [
      ("latent", Diff);
    ] in
    let rvars_disc = [ ] in
    { rte_name      = "vae (1-model)";
      rte_filename  = "../srepar/srepar/examples/vae/model.py";
      rte_inlinefns = ["main"];
      rte_diffinfo  = Some (params @ rvars);
      rte_rvars      = List.map fst rvars;
      rte_rvars_disc = rvars_disc;
      rte_guide_pars = List.map fst params; }
   );
   (
    let params = [
      ("encoder_fc1",  Diff);
      ("encoder_fc21", Diff);
      ("encoder_fc22", Diff);
    ] in
    let rvars  = [
      ("latent", Diff);
    ] in
    let rvars_disc = [ ] in
    { rte_name      = "vae (2-guide)";
      rte_filename  = "../srepar/srepar/examples/vae/guide.py";
      rte_inlinefns = ["main"];
      rte_diffinfo  = Some (params @ rvars);
      rte_rvars      = List.map fst rvars;
      rte_rvars_disc = rvars_disc;
      rte_guide_pars = List.map fst params; }
   );

   (*************************)
   (* Examples for POPL'23. *)
   (*************************)
   (* spnor *)
   (
    let params = [
    ] in
    let rvars = [
      ("z1", Diff);
      ("z2", Top); (* used in if condition *)
    ] in
    let rvars_disc = [ ] in
    { rte_name      = "spnor (1-model)";
      rte_filename  = "../srepar/srepar/examples/spnor/model.py";
      rte_inlinefns = ["model"];
      rte_diffinfo  = Some (params @ rvars);
      rte_rvars      = List.map fst rvars;
      rte_rvars_disc = rvars_disc;
      rte_guide_pars = List.map fst params; }
   );
   (
    let params = [
      ("theta1", Diff);
      ("theta2", Diff);
    ] in
    let rvars = [
      ("z1", Diff);
      ("z2", Diff);
    ] in
    let rvars_disc = [ ] in
    { rte_name      = "spnor (2-guide)";
      rte_filename  = "../srepar/srepar/examples/spnor/guide.py";
      rte_inlinefns = ["guide"];
      rte_diffinfo  = Some (params @ rvars);
      rte_rvars      = List.map fst rvars;
      rte_rvars_disc = rvars_disc;
      rte_guide_pars = List.map fst params; }
   );

   (* dpmm *)
   (
    let params = [] in
    let rvars = [
      ("beta",   Diff);
      ("lambda", Diff);
      ("z",      Top);  (* discrete distribution. *)
    ] in
    let rvars_disc = [ "z" ] in
    { rte_name      = "dpmm (1-model)";
      rte_filename  = "../srepar/srepar/examples/dpmm/model.py";
      rte_inlinefns = ["main"];
      rte_diffinfo  = Some (params @ rvars);
      rte_rvars      = List.map fst rvars;
      rte_rvars_disc = rvars_disc;
      rte_guide_pars = List.map fst params; }
   );
   (
    let params = [
      ("kappa", Diff);
      ("tau_0", Diff);
      ("tau_1", Diff);
      ("phi",   Diff);
    ] in
    let rvars = [
      ("beta",   Diff);
      ("lambda", Diff);
      ("z",      Top);  (* discrete distribution. *)
    ] in
    let rvars_disc = [ "z" ] in
    { rte_name      = "dpmm (2-guide)";
      rte_filename  = "../srepar/srepar/examples/dpmm/guide.py";
      rte_inlinefns = ["main"];
      rte_diffinfo  = Some (params @ rvars);
      rte_rvars      = List.map fst rvars;
      rte_rvars_disc = rvars_disc;
      rte_guide_pars = List.map fst params; }
   );

   (* cvae *)
   (
    let params = [
      ("prior_net_fc1",  Lips); (* applied to nn.ReLU()(...) *)
      ("prior_net_fc2",  Lips); (* applied to nn.ReLU()(...) *)
      ("prior_net_fc31", Diff);
      ("prior_net_fc32", Diff);
      ("gener_net_fc1",  Lips); (* applied to nn.ReLU()(...) *)
      ("gener_net_fc2",  Lips); (* applied to nn.ReLU()(...) *)
      ("gener_net_fc3",  Diff);
    ] in
    let rvars = [
      ("z", Lips); (* applied to nn.ReLU()(...) *)
    ] in
    let rvars_disc = [ ] in
    { rte_name      = "cvae (1-model)";
      rte_filename  = "../srepar/srepar/examples/cvae/modelguide.py";
      rte_inlinefns = ["model"]; (* inline the functions in `ret_inlinefns` *)
      rte_diffinfo  = Some (params @ rvars);
      rte_rvars      = List.map fst rvars;
      rte_rvars_disc = rvars_disc;
      rte_guide_pars = List.map fst params; }
   );
   (
    let params = [
      ("prior_net_fc1",  Lips); (* applied to nn.ReLU()(...) *)
      ("prior_net_fc2",  Lips); (* applied to nn.ReLU()(...) *)
      ("prior_net_fc31", Diff);
      ("prior_net_fc32", Diff);
      ("recog_net_fc1",  Lips); (* applied to nn.ReLU()(...) *)
      ("recog_net_fc2",  Lips); (* applied to nn.ReLU()(...) *)
      ("recog_net_fc31", Diff);
      ("recog_net_fc32", Diff);
    ] in
    let rvars = [
      ("z", Diff);
    ] in
    let rvars_disc = [ ] in
    { rte_name      = "cvae (2-guide)";
      rte_filename  = "../srepar/srepar/examples/cvae/modelguide.py";
      rte_inlinefns = ["guide"];
      rte_diffinfo  = Some (params @ rvars);
      rte_rvars      = List.map fst rvars;
      rte_rvars_disc = rvars_disc;
      rte_guide_pars = List.map fst params; }
   );

   (* scanvi *)
   (
    let params = [
      ("z2_decoder_fc_1", Lips); (* applied to nn.ReLU()(...) *)
      ("z2_decoder_fc_2", Lips); (* applied to nn.ReLU()(...) *)
      ("z2_decoder_fc_4", Diff);
      ("z2_decoder_fc_5", Diff);
      ("x_decoder_fc_1",  Lips); (* applied to nn.ReLU()(...) *)
      ("x_decoder_fc_2",  Lips); (* applied to nn.ReLU()(...) *)
      ("x_decoder_fc_4",  Diff);
      ("x_decoder_fc_5",  Diff);
      ("inverse_dispersion", Diff);
    ] in
    let rvars = [
      ("z1", Lips); (* applied to nn.ReLU()(...) *)
      ("z2", Lips); (* applied to nn.ReLU()(...) *)
      ("l",  Diff);
      ("y" , Top ); (* discrete distribution. *)
    ] in
    let rvars_disc = [ "y" ] in
    { rte_name      = "scanvi (1-model)";
      rte_filename  = "../srepar/srepar/examples/scanvi/modelguide.py";
      rte_inlinefns = ["model"];
      rte_diffinfo  = Some (params @ rvars);
      rte_rvars      = List.map fst rvars;
      rte_rvars_disc = rvars_disc;
      rte_guide_pars = List.map fst params; }
   );
   (
    let params = [
      ("z2l_encoder_fc_1", Lips); (* applied to nn.ReLU()(...) *)
      ("z2l_encoder_fc_2", Lips); (* applied to nn.ReLU()(...) *)
      ("z2l_encoder_fc_4", Diff);
      ("z2l_encoder_fc_5", Diff);
      ("z1_encoder_fc_1",  Lips); (* applied to nn.ReLU()(...) *)
      ("z1_encoder_fc_2",  Lips); (* applied to nn.ReLU()(...) *)
      ("z1_encoder_fc_4",  Diff);
      ("z1_encoder_fc_5",  Diff);
      ("classifier_fc_1",  Lips); (* applied to nn.ReLU()(...) *)
      ("classifier_fc_2",  Lips); (* applied to nn.ReLU()(...) *)
      ("classifier_fc_4",  Diff);
      ("classifier_fc_5",  Diff);
    ] in
    let rvars = [
      ("z1", Diff);
      ("z2", Lips); (* applied to nn.ReLU()(...) *)
      ("l",  Diff);
      ("y" , Top ); (* discrete distribution. *)
    ] in
    let rvars_disc = [ "y" ] in
    { rte_name      = "scanvi (2-guide)";
      rte_filename  = "../srepar/srepar/examples/scanvi/modelguide.py";
      rte_inlinefns = ["guide"];
      rte_diffinfo  = Some (params @ rvars);
      rte_rvars      = List.map fst rvars;
      rte_rvars_disc = rvars_disc;
      rte_guide_pars = List.map fst params; }
   );

   (* prodlda *)
   (
    let params = [
      ("decoder_beta", Diff);
    ] in
    let rvars = [
      ("logtheta", Diff);
    ] in
    let rvars_disc = [ ] in
    { rte_name       = "prodlda (1-model)";
      rte_filename   = "../srepar/srepar/examples/prodlda/modelguide.py";
      rte_inlinefns  = ["model"];
      rte_diffinfo   = Some (params @ rvars);
      rte_rvars      = List.map fst rvars;
      rte_rvars_disc = rvars_disc;
      rte_guide_pars = List.map fst params; }
   );
   (
    let params = [
      ("encoder_fc1",  Diff);
      ("encoder_fc2",  Diff);
      ("encoder_fcmu", Diff);
      ("encoder_fclv", Diff);
    ] in
    let rvars = [
      ("logtheta", Diff);
    ] in
    let rvars_disc = [ ] in
    { rte_name       = "prodlda (2-guide)";
      rte_filename   = "../srepar/srepar/examples/prodlda/modelguide.py";
      rte_inlinefns  = ["guide"];
      rte_diffinfo   = Some (params @ rvars);
      rte_rvars      = List.map fst rvars;
      rte_rvars_disc = rvars_disc;
      rte_guide_pars = List.map fst params; }
   );

   (* mhmm *)
   (
    let params = [
      ("step_zi_param",              Diff);
      ("step_param_concentration",   Diff);
      ("step_param_rate",            Diff);
      ("angle_param_concentration",  Diff);
      ("angle_param_loc",            Diff);
      ("omega_zi_param",             Diff);
      ("omega_param_concentration0", Diff);
      ("omega_param_concentration1", Diff);
    ] in
    let rvars = [
      ("eps_g",      Diff);
      ("eps_i",      Diff);
      ("y_{}" ,      Top); (* discrete distribution. *)
    ] in
    let rvars_disc = [ "y_{}" ] in
    { rte_name      = "mhmm (1-model)";
      rte_filename  = "../srepar/srepar/examples/mhmm/modelguide.py";
      rte_inlinefns = ["model"];
      rte_diffinfo  = Some (params @ rvars);
      rte_rvars      = List.map fst rvars;
      rte_rvars_disc = rvars_disc;
      rte_guide_pars = List.map fst params; }
   );
   (
    let params = [
      ("loc_group",        Diff);
      ("scale_group",      Diff);
      ("loc_individual",   Diff);
      ("scale_individual", Diff);
    ] in
    let rvars = [
      ("eps_g", Diff);
      ("eps_i", Diff);
    ] in
    let rvars_disc = [ ] in
    { rte_name      = "mhmm (2-guide)";
      rte_filename  = "../srepar/srepar/examples/mhmm/modelguide.py";
      rte_inlinefns = ["guide"];
      rte_diffinfo  = Some (params @ rvars);
      rte_rvars      = List.map fst rvars;
      rte_rvars_disc = rvars_disc;
      rte_guide_pars = List.map fst params; }
   );
 ]
let tests_diff_run = ref tests_diff_all


(** Result (for the final log) *)
type result =
    { res_name:    string ;
      res_pass:    bool ;
      res_time:    float ;
      res_comment: string }
type results = result SM.t
let fp_results (fmt: form) (r: results): unit =
  let cl s =
    let nmax = 100 in
    if String.length s < nmax then s
    else String.sub s 0 nmax in
  SM.iter
    (fun n r ->
      let len = min (String.length n) 18 in
      F.fprintf fmt "%s%s %s %.4fs   %s\n" n (String.make (20-len) ' ')
        (if r.res_pass then "OK  " else "FAIL") r.res_time (cl r.res_comment)
    ) r


(** Reg-test one example *)
let do_regtest (verb: bool) dnum
    (dp_goal: diff_prop)
    ~(flag_fwd: bool)
    ~(flag_comp: bool)
    ~(flag_old_comp: bool)
    ~(flag_pyast: bool)
    (c_crash, c_ok, c_ko, c_unk, c_ko_rts, results) (rt: rt_entry)
    : int * int * int * int * (rt_entry list) * results =
  let ddiff_expected =
    let map_fun ((s,dp): string * diff_prop): string list =
      if diff_prop_leq dp dp_goal then [s] else [] in
    let option_fun (sdp_list: (string * diff_prop) list): string list =
      List.flatten (List.map map_fun sdp_list) in
    option_map option_fun rt.rte_diffinfo in
  let time_0 = Unix.gettimeofday () in
  let ddiff_analyzed, rcom =
    try
      let a =
        analyze dnum dp_goal ~title:"" ~flag_fwd ~flag_comp ~flag_old_comp
          ~flag_pyast ~inline_fns:rt.rte_inlinefns
          ~input:(Either.Left rt.rte_filename) verb in
      Some a, ""
    with
    | Apron.Manager.Error s ->
        F.printf "Apron error: %a\n" Apron.Manager.print_exclog s;
        None, "Apron error"
    | e ->
        F.printf "Exception: %s\n" (Printexc.to_string e);
        None, F.asprintf "Exc: %s" (Printexc.to_string e) in
  let time_1 = Unix.gettimeofday () in
  let delta = time_1 -. time_0 in
  let res = { res_name    = rt.rte_name;
              res_pass    = (ddiff_analyzed != None);
              res_time    = delta;
              res_comment = rcom } in
  let c_crash, c_ok, c_ko, c_unk, c_ko_rts, res =
    match ddiff_expected, ddiff_analyzed with
    | _, None ->
        (* the analysis crashed *)
        c_crash + 1, c_ok, c_ko, c_unk, c_ko_rts, res
    | Some l, Some di ->
        let s_res = Ddom_diff.diff_info_gen_dens_diff di in
        (* some expected result is known *)
        let s_norm = List.fold_left (fun a i -> SS.add i a) SS.empty l in
        F.printf "RESULT FOR %s:\noracle:\t%a\nresult:\t%a\n"
          rt.rte_filename ss_fp s_norm ss_fp s_res;
        let res =
          let rvarpars_all_set =
            match rt.rte_diffinfo with
            | Some l -> SS.of_list (List.map fst l)
            | None   -> SS.empty in
          let rvarpars_disc_set = SS.of_list rt.rte_rvars_disc in
          let rvarpars_cont_set = SS.diff rvarpars_all_set rvarpars_disc_set in
          let totl_cont = SS.cardinal rvarpars_cont_set in
          let totl_disc = SS.cardinal rvarpars_disc_set in
          let manl_d    = SS.cardinal (SS.of_list l) in
          let manl_nd   = (totl_cont + totl_disc) - manl_d in
          let ours_d    = SS.cardinal (Ddom_diff.diff_info_gen_dens_diff di) in
          let ours_nd   = SS.cardinal (Ddom_diff.diff_info_gen_dens_ndiff di) in
          let msg =
            F.asprintf "#RP:  manl(smt, ~smt) %2d %2d | ours(smt, may~smt) %2d %2d | totl(cont, disc) %2d %2d"
              manl_d manl_nd ours_d ours_nd totl_cont totl_disc in
          { res with
            res_comment = msg } in
        if SS.equal s_norm s_res then
          (* the analysis result coincides with the expected result *)
          c_crash, c_ok + 1, c_ko, c_unk, c_ko_rts, res
        else
          (* the analysis result does not coincide with the expected result *)
          let m =
            F.asprintf "%s, diff, exp %d diff" res.res_comment (SS.cardinal s_norm) in
          let res = { res with
                      res_pass    = false ;
                      res_comment = m } in
          c_crash, c_ok, c_ko + 1, c_unk, (rt :: c_ko_rts), res
    | None, Some _ ->
        (* the analysis completes, but expected result was not provided *)
        let res = { res with res_comment = "unknown outcome" } in
        c_crash, c_ok, c_ko, c_unk + 1, c_ko_rts, res in
  c_crash, c_ok, c_ko, c_unk + 1, c_ko_rts, SM.add rt.rte_name res results

(** Reg-test all examples *)
let do_regtests (verb: bool) dnum (dp_goal: diff_prop)
    ~(flag_fwd: bool) ~(flag_comp: bool) ~(flag_old_comp: bool)
    ~(flag_pyast: bool)
    : unit =
  let sep = String.make 78 '=' in
  let c_tot = List.length !tests_diff_run in
  let c_crash, c_ok, c_ko, c_unk, c_ko_rts_rev, res =
    List.fold_left
      (do_regtest verb dnum dp_goal ~flag_fwd ~flag_comp ~flag_old_comp ~flag_pyast)
      (0, 0, 0, 0, [], SM.empty) !tests_diff_run in
  let c_ko_rts = List.rev c_ko_rts_rev in
  F.printf "%s\nREGRESSION TEST RESULTS:\n%s\n" sep sep;
  F.printf "Total:   %d\n" c_tot;
  F.printf "OK:      %d\n" c_ok;
  F.printf "KO:      %d\n" c_ko;
  F.printf "Crash:   %d\n" c_crash;
  F.printf "Unknown: %d\n" c_unk;
  F.printf "Failed cases:\n";
  List.iter (fun rt -> F.printf "\t%s\n" rt.rte_name) c_ko_rts;
  F.printf "%s\nDetailed results:\n%s\n%a%s\n" sep sep fp_results res sep


(** Apply selective reparam for one example *)
let apply_srepar (_fin_name: string) (main_name: string) (_dndiff: SS.t): unit =
  let aux (fout_name_suffix: string) (dndiff_str: string): unit =
    begin
      (* python codes to inject/match *)
      let pp_preamble chan () = (* for import srepar *)
        F.fprintf chan "'''\n";
        F.fprintf chan "Auto-generated by `whitebox/refact/batch_diff/batch_diff.ml`.\n";
        F.fprintf chan "'''\n";
        F.fprintf chan "from srepar.lib.srepar import set_no_reparam_names\n\n" in
      let pp_srepar chan (s: string) = (* for selective reparam *)
        F.fprintf chan "    # set up reparameterisation\n";
        F.fprintf chan "    set_no_reparam_names(%s)\n\n" s in
      let re_main = (* to be matched as the start of main func *)
        Str.regexp (F.sprintf "def[ ]+%s[ ]*(.*)[ ]*:[ ]*$" main_name) in

      (* fin_name, fout_name_tmp, fout_name *)
      let _dirname        = Filename.dirname  _fin_name in
      let _fin_name_extrm = Filename.basename _fin_name |> Filename.remove_extension in
      let _fin_name_ext   = Filename.basename _fin_name |> Filename.extension in
      let fout_name_tmp   = Filename.concat _dirname "tmp" in
      let fout_name       =
        Filename.concat _dirname
          (F.asprintf "%s_%s%s" _fin_name_extrm fout_name_suffix _fin_name_ext) in
      let fin_name        =
        if not (_fin_name_extrm = "modelguide" && main_name = "guide")
        then _fin_name (* normal case. *)
        else fout_name (* special case. inject codes to fout_name. *) in

      (* open inchan, outchan. *)
      let fin      = Unix.openfile fin_name [ Unix.O_RDONLY ] 0o644 in
      let fout_tmp = Unix.openfile fout_name_tmp [ Unix.O_WRONLY; Unix.O_CREAT; Unix.O_TRUNC ] 0o644 in
      let inchan   = Unix.in_channel_of_descr  fin  in
      let outchan  = Unix.out_channel_of_descr fout_tmp in
      let outfmt   = F.formatter_of_out_channel outchan in

      (* inject python codes. fin ---> fout_name_tmp. *)
      F.fprintf outfmt "%a" pp_preamble ();
      let pp_srepar_done = ref false in
      begin
        try
          while true do
            let line = input_line inchan in
            F.fprintf outfmt "%s\n" line;
            if (not !pp_srepar_done) && (Str.string_match re_main line 0) then
              ( pp_srepar_done := true;
                F.fprintf outfmt "%a" pp_srepar dndiff_str )
          done
        with End_of_file -> assert(!pp_srepar_done)
      end;

      (* close outchan and rename. fout_name_tmp ---> fout_name. *)
      close_out outchan;
      Unix.rename fout_name_tmp fout_name
    end in

  (* dndiff_str *)
  let dndiff =
    (* special handle for air/{model,guide}.py *)
    if (Filename.dirname _fin_name) = "../srepar/srepar/examples/air"
    then (F.printf "  NOTE: 'z_where_{}' is manually removed from non-reparam'l params";
          F.printf " when python codes are generated!\n";
          SS.diff _dndiff (SS.singleton "z_where_{}"))
    else _dndiff in
  let dndiff_str =
    F.asprintf "[%s]"
      ((buf_to_string (buf_list ", " buf_string)) (SS.elements dndiff)) in

  (* inject python codes *)
  aux "ours"  dndiff_str;
  aux "score" "True";
  aux "repar" "[]"

(** Apply selective reparam for all examples *)
let apply_srepars dnum (dp_goal: diff_prop)
    ~(flag_fwd: bool) ~(flag_comp: bool) ~(flag_old_comp: bool)
    ~(flag_pyast: bool): unit =
  let rec aux (tests: rt_table): unit =
    match tests with
    | rt1 :: rt2 :: tests_tl ->
       begin
         (* analyze *)
         let time_0 = Unix.gettimeofday () in
         let dndiff1 =
           let di =
             analyze dnum dp_goal ~title:"(model)"
               ~flag_fwd ~flag_comp ~flag_old_comp
               ~flag_pyast ~inline_fns:rt1.rte_inlinefns
               ~input:(Either.Left rt1.rte_filename) false in
           Ddom_diff.diff_info_gen_dens_diff di in
         let time_1 = Unix.gettimeofday () in
         let dndiff2 =
           let di =
             analyze dnum dp_goal ~title:"(guide)"
               ~flag_fwd ~flag_comp ~flag_old_comp
               ~flag_pyast ~inline_fns:rt2.rte_inlinefns
               ~input:(Either.Left rt2.rte_filename) false in
           Ddom_diff.diff_info_gen_dens_diff di in
         let time_2 = Unix.gettimeofday () in
         let dndiff = SS.union dndiff1 dndiff2 in
         F.printf "%-35s\t=>\t%a\n"
           (Filename.dirname rt1.rte_filename) ss_fp dndiff;
         (* apply srepar *)
         let time_3 = Unix.gettimeofday () in
         apply_srepar rt1.rte_filename (List.hd rt1.rte_inlinefns) dndiff;
         let time_4 = Unix.gettimeofday () in
         apply_srepar rt2.rte_filename (List.hd rt2.rte_inlinefns) dndiff;
         let time_5 = Unix.gettimeofday () in
         F.printf "TIME,%s,model: %.4f s\n" rt1.rte_filename
           (time_1 -. time_0 +. time_4 -. time_3);
         F.printf "TIME,%s,guide: %.4f s\n" rt2.rte_filename
           (time_2 -. time_1 +. time_5 -. time_4);
         aux tests_tl
       end
    | [] -> ()
    | _ -> failwith "apply_repar: tests_diff has an odd number of entries" in
  let sep = String.make 78 '=' in
  F.printf "\n\n";
  F.printf "Generating selectively reparameterised models/guides...\n\n";
  F.printf "%s\nNON-REPARAMETERISABLE PARAMETERS\n%s\n" sep sep;
  aux !tests_diff_run


let apply_srepars_new dnum (dp_goal: diff_prop)
    ~(flag_fwd: bool) ~(flag_comp: bool) ~(flag_old_comp: bool)
    ~(flag_pyast: bool): unit =
  let sep = String.make 78 '=' ^ "\n" in
  let rec aux (tests: rt_table) (acc: results): results =
    match tests with
    | rtm :: rtg :: tests_tl ->
        F.printf "%sGenerating selectively reparameterised models/guides...\n%s" sep sep;
        (* general settings and data *)
        let f_guide = rtg.rte_filename and f_model = rtm.rte_filename in
        F.printf "Model: %s\nGuide: %s\n%s" f_model f_guide sep;
        F.printf "NumParams(Model+Guide): %d (%s)\n%s" ((List.length rtm.rte_guide_pars) + (List.length rtg.rte_guide_pars)) f_model sep;
        let time_start = Unix.gettimeofday () in
        let ir_model =
          Ir_parse.output := false ;
          Ir_parse.parse_code ~use_pyast:flag_pyast ~inline_fns:rtm.rte_inlinefns
            None (AI_pyfile f_model)
            |> Ir_util.simplify_delta_prog in
        let ir_guide =
          Ir_parse.output := false ;
          Ir_parse.parse_code ~use_pyast:flag_pyast ~inline_fns:rtg.rte_inlinefns
            None (AI_pyfile f_guide)
            |> Ir_util.simplify_delta_prog in
        (* analyze model and guide *)
        let time_0 = Unix.gettimeofday () in
        let di_model =
          analyze dnum dp_goal ~title:"(model)"
            ~flag_fwd ~flag_comp ~flag_old_comp
            ~flag_pyast ~inline_fns:rtm.rte_inlinefns
            ~input:(Either.Right (f_model, ir_model)) false in
        let time_1 = Unix.gettimeofday () in
        let di_guide =
          analyze dnum dp_goal ~title:"(guide)"
            ~flag_fwd ~flag_comp ~flag_old_comp
            ~flag_pyast ~inline_fns:rtg.rte_inlinefns
            ~input:(Either.Right (f_guide, ir_guide)) false in
        let time_2 = Unix.gettimeofday () in
        (* condition for reparameterisation *)
        let theta = SS.of_list (rtm.rte_guide_pars @ rtg.rte_guide_pars) in
        let allrvars = SS.of_list (rtm.rte_rvars @ rtg.rte_rvars) in
        let allpars = SS.union theta allrvars in
        let ndiff =
          let f di a = SM.fold (fun _ s -> SS.union s) di.di_prb_ndiff a in
          let ndiff = f di_model di_model.di_dens_ndiff in
          let ndiff = f di_guide ndiff in
          ndiff in
        let k = SS.diff allpars ndiff in
        let condition = SS.inter ndiff theta = SS.empty in
        let sep_1 = String.make 78 '=' in
        let sep_2 = String.make 78 '-' in
        F.printf "%s\nREPARAMETERISATION CONDITIONS:\n%s\n" sep_1 sep_1;
        F.printf "Theta <= K ?       = %b\n" (SS.inter ndiff theta = SS.empty);
        F.printf "Allpars (%d elts)  = %a\n" (SS.cardinal allpars) ss_fp allpars;
        F.printf "Theta (%d elts)    = %a\n" (SS.cardinal theta) ss_fp theta;
        F.printf "K (%d elts)        = %a\n" (SS.cardinal k) ss_fp k;
        F.printf "Theta <= allpars ? = %b\n" (SS.subset theta allpars);
        F.printf "ND (%d elts)       = %a\n" (SS.cardinal ndiff) ss_fp ndiff;
        F.printf "%s\nSets of reparameterised variables\n%s\n" sep_2 sep_2;
        (* transformation of model and guide *)
        F.printf "Repar set: %a\n" ss_fp k;
        let repar_model = Reparam.sel_repar k ir_model in
        F.printf "%s\nReparameterised model:\n%s\n%a%s\n" sep_2 sep_2 Ir_util.fp_prog repar_model sep_2;
        let repar_guide = Reparam.sel_repar k ir_guide in
        F.printf "Reparameterised guide:\n%s\n%a%s\n" sep_2 Ir_util.fp_prog repar_guide sep_1;
        (* analyze reparameterised model and guide *)
        (*let time_3 = Unix.gettimeofday () in
        let di_repar_model =
          analyze dnum dp_goal ~flag_fwd ~flag_comp ~flag_old_comp
            ~flag_pyast ~inline_fns:rtm.rte_inlinefns
            ~input:(Either.Right (f_model, repar_model)) false in*)
        let time_4 = Unix.gettimeofday () in
        let di_repar_guide =
          analyze dnum dp_goal ~title:"(reparameterized guide)"
            ~flag_fwd ~flag_comp ~flag_old_comp
            ~flag_pyast ~inline_fns:rtg.rte_inlinefns
            ~input:(Either.Right (f_guide, repar_guide)) false in
        let time_5 = Unix.gettimeofday () in
        (* check stabilisation conditions *)
        let check_condition msg di =
          let ldebug = false in
          if ldebug then
            F.printf "Checking condition for %s\n%a" msg
              (Ddom_diff.diff_info_fpi "  ") di;
          let s = SM.fold (fun x -> SS.union) di.di_val_ndiff SS.empty in
          (* WL: edited the below part for `s`, to reflect the change in
           *     our algorithm in Section 6. *)
          (* let s =
            SS.fold
              (fun v acc ->
                try SS.union (SM.find v di.di_prb_ndiff) acc
                with Not_found -> acc
              ) k s in *)
          let s = SM.fold (fun x -> SS.union) di.di_prb_ndiff s in
          let inter = SS.inter s theta in
          let condition = inter = SS.empty in
          if ldebug then
            F.printf "Condition on %s: %b\n theta: %a\n s:     %a\n inter: %a\n"
              msg (inter = SS.empty) ss_fp theta ss_fp s ss_fp inter;
          condition in
        (*let condition_model = check_condition "model" di_repar_model in*)
        let condition_guide = check_condition "guide" di_repar_guide in

        (* show timing information *)
        let time_end = Unix.gettimeofday () in
        let elapsed = time_end -. time_start in
        let time_model = time_1 -. time_0 (* +. time_4 -. time_3 *)
        and time_guide = time_2 -. time_1 +. time_5 -. time_4 in
        F.printf "TIME,%s,model: %.4f s\n" rtm.rte_filename time_model;
        F.printf "TIME,%s,guide: %.4f s\n" rtg.rte_filename time_guide;

        (* results for both model and guide *)
        let rvars_all_set  = allrvars in
        let rvars_disc_set = SS.of_list (rtm.rte_rvars_disc @ rtg.rte_rvars_disc) in
        let rvars_cont_set = SS.diff rvars_all_set rvars_disc_set in
        let totl_cont = SS.cardinal rvars_cont_set in
        let totl_disc = SS.cardinal rvars_disc_set in
        let ours_snd  = SS.cardinal (SS.inter rvars_all_set k) in
        let ours_nsnd_cont_set = SS.diff rvars_cont_set k in
        let msg =
          F.asprintf "#R:  ours(sound) %2d | totl(cont, disc) %2d %2d | ours(may~sound && cont) %a"
            ours_snd totl_cont totl_disc ss_fp ours_nsnd_cont_set in
        let res_name = Str.global_replace (Str.regexp "(2-guide)") "" rtg.rte_name in
        (* let res_model = { res_name    = rtm.rte_name;
         *                   res_pass    = condition && condition_model;
         *                   res_time    = time_model;
         *                   res_comment = msg } in *)
        let res_guide = { res_name    = res_name;
                          res_pass    = condition && condition_guide;
                          res_time    = (*time_guide +. time_model*)elapsed;
                          res_comment = msg } in
        aux tests_tl
          ((*SM.add rtm.rte_name res_model*)
            (SM.add res_name res_guide acc))
    | [] -> acc
    | _ -> failwith "apply_repar: tests_diff has an odd number of entries" in
  let results = aux !tests_diff_run SM.empty in
  let sep = String.make 78 '=' in
  F.printf "%s\nREGRESSION TEST RESULTS:\n%s\n" sep sep;
  F.printf "%a%s\n" fp_results results sep


(** Function to iterate the analysis *)
let main () =
  let fverb = ref false in
  let dnum: ad_num ref         = ref AD_sgn in
  let goal: diff_prop ref      = ref Lips
  and flag_fwd: bool ref       = ref false
  and flag_comp: bool ref      = ref true
  and flag_old_comp: bool ref  = ref false
  and flag_pyast: bool ref     = ref true
  and flag_srepar: bool ref    = ref true
  and flag_srepar_o: bool ref  = ref false
  and flag_srepar_n: bool ref  = ref false
  and show_list: bool ref      = ref false
  and sel_test: int option ref = ref None in
  let dnum_set v = Arg.Unit (fun () -> dnum := v) in
  let goal_set g = Arg.Unit (fun () -> goal := g) in
  let set_sel i = sel_test := Some i in
  Arg.parse
    [ (* Numerical domain *)
      "-ai-box",   dnum_set AD_box,    "Num analysis, Apron, Boxes" ;
      "-ai-oct",   dnum_set AD_oct,    "Num analysis, Apron, Octagons" ;
      "-ai-pol",   dnum_set AD_pol,    "Num analysis, Apron, Polyhedra" ;
      "-ai-sgn",   dnum_set AD_sgn,    "Num analysis, Basic, Signs" ;
      (* Target smoothness property *)
      "-g-lipsch", goal_set Lips,      "Goal set to Lipschitz (default)" ;
      "-g-diff",   goal_set Diff,      "Goal set to Differentiability" ;
      (* Forward analysis on/off *)
      "-no-fwd",   Arg.Clear flag_fwd,   "Forward analysis OFF (default)" ;
      "-fwd",      Arg.Set flag_fwd,     "Forward analysis ON" ;
      (* Compositional analysis on/off *)
      "-no-comp",  Arg.Clear flag_comp,  "Compositional analysis OFF" ;
      "-comp",     Arg.Set flag_comp,    "Compositional analysis ON (default)" ;
      "-old-comp", Arg.Set flag_old_comp,"Compositional analysis old representation" ;
      (* Verbosity on or off *)
      "-verb",     Arg.Set fverb,      "Verbose mode" ;
      (* Use of the pyast IR *)
      "-no-pyast", Arg.Clear flag_pyast, "Uses old Pyast AST" ;
      "-pyast",    Arg.Set flag_pyast,   "Uses external Pyast AST" ;
      (* Verbosity *)
      "-v",        Arg.Set fverb,      "Verbose mode" ;
      (* List of examples and selection of a single test (for debug) *)
      "-l",        Arg.Set show_list,  "Show list of tests" ;
      "-s",        Arg.Int set_sel,    "Select one single to test to run" ;
      "-sel",      Arg.Int set_sel,    "Select one single to test to run" ;
      (* Selects execution mode *)
      "-analysis", Arg.Clear flag_srepar, "Do only analysis (default)" ;
      "-srepar-o", Arg.Set flag_srepar_o, "Do analysis + reparameterisation (old)" ;
      "-srepar-n", Arg.Set flag_srepar_n, "New selective reparameterisation" ;
    ]
    (fun s -> failwith (F.asprintf "unbound %S" s))
    "Differentiability analysis, batch mode";
  (* Optional display of list of tests, and exit *)
  if !show_list then
    begin
      F.printf "List of tests:\n";
      List.iteri (fun i rte -> F.printf " %.2d: %s\n" i rte.rte_filename)
        !tests_diff_run;
      exit 0
    end;
  (* Optional selection of a single test to run (for debug) *)
  begin
    match !sel_test with
    | None -> ( )
    | Some i ->
        let i = if i mod 2 = 0 then i else i - 1 in
        let f j _ = j = i || j = i + 1 in
        tests_diff_run := List.filteri f !tests_diff_run
  end;
  if !flag_srepar && !flag_srepar_n then
    (* New selective reparam testing. *)
    apply_srepars_new !dnum !goal
      ~flag_fwd:!flag_fwd ~flag_comp:!flag_comp ~flag_old_comp:!flag_old_comp
      ~flag_pyast:!flag_pyast
  else if !flag_srepar && !flag_srepar_o then
    (* Apply selective reparam. *)
    begin
      try
        apply_srepars !dnum !goal
          ~flag_fwd:!flag_fwd ~flag_comp:!flag_comp ~flag_old_comp:!flag_old_comp
          ~flag_pyast:!flag_pyast
      with e ->
        F.printf "Exception in the reparam stage: %s\n" (Printexc.to_string e)
    end
  else
    (* Only run the analysis regression tests *)
    do_regtests !fverb !dnum !goal
      ~flag_fwd:!flag_fwd ~flag_comp:!flag_comp ~flag_old_comp:!flag_old_comp
      ~flag_pyast:!flag_pyast;
  F.printf "Batch testing finished!\n"

(** Start *)
let _ = ignore (main ())
