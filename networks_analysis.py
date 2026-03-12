import torch
import dill
import click
import os
import pathlib  
import hydra
import json

from diffusion_policy.workspace.base_workspace import BaseWorkspace

def compute_kl(mu, sigma, w0, prior_sigma):
    """
    KL( N(mu, sigma^2) || N(w0, prior_sigma^2) )
    """
    var_q = sigma**2
    var_p = prior_sigma**2

    kl = 0.5 * (
        torch.log(var_p / var_q)
        + (var_q + (mu - w0)**2) / var_p
        - 1.0
    )
    return kl.sum()

def bayesian_uncertainty_report(dp_model, bayes_model):
    report = {}
    for k, v in bayes_model.state_dict().items():
        if k.endswith(".weight.rho"):
            weight_rho = v.flatten()
            sigma = torch.log(1 + torch.exp(weight_rho))
            mean_sigma = sigma.mean().item()
            mu_parameter_name = k.replace(".weight.rho", ".weight.mu")
            mu = bayes_model.state_dict()[mu_parameter_name].flatten()
            w0 = dp_model.state_dict()[k.replace(".weight.rho", ".weight")].flatten()
            kl = compute_kl(mu, sigma, w0, prior_sigma=1e-2)

            report[k] = {
                "mean_rho": weight_rho.mean().item(),
                "mean_sigma": mean_sigma,
                "snr": float(mu.norm().item() / (sigma.norm() + 1e-12).item()),
                #"shape": weight_rho.shape,
                #"kl": kl.item()
            }

        elif k.endswith(".bias.rho"):
            bias_rho = v.flatten()
            sigma = torch.log(1 + torch.exp(bias_rho))
            mean_sigma = sigma.mean().item()
            mu_parameter_name = k.replace(".bias.rho", ".bias.mu")
            mu = bayes_model.state_dict()[mu_parameter_name].flatten()
            w0 = dp_model.state_dict()[k.replace(".bias.rho", ".bias")].flatten()
            kl = compute_kl(mu, sigma, w0, prior_sigma=1e-2)

            report[k] = {
                "mean_rho": bias_rho.mean().item(),
                "mean_sigma": mean_sigma,
                "snr": float(mu.norm().item() / (sigma.norm() + 1e-12).item()),
                #"shape": bias_rho.shape,
                #"kl": kl.item()
            }
    return report

def compare_mean_to_baseline(DP_model, Bayes_model):
    report = {}

    # Extract non-scalar parameters from DP_model
    dp_params = {
    k: v
    for k, v in DP_model.state_dict().items()
    if v.ndim > 0  # skip scalars like num_batches_tracked
    }
    print (f"Extracted {len(dp_params)} non-scalar parameters from DP_model")

    for k, v in dp_params.items():
        print (f"DP_model parameter: {k} with shape {v.shape}" )
    # Extract mean parameters from Bayes_model
    bayes_params = {}
    for k, v in Bayes_model.state_dict().items():
        print (f"Processing parameter: {k} with shape {v.shape}" )
        if k.endswith(".rho") or k.endswith(".weight_prior.mu") or k.endswith(".bias_prior.mu"):
            #print (f"Skipping parameter: {k}")
            continue  # skip rho parameters
        if k.endswith(".weight.mu"):
            #print (f"Mapping parameter: {k} to {k.replace('.weight.mu', '.weight')}")
            bayes_params[k.replace(".weight.mu", ".weight")] = v
        elif k.endswith(".bias.mu"):
            #print (f"Mapping parameter: {k} to {k.replace('.bias.mu', '.bias')}")
            bayes_params[k.replace(".bias.mu", ".bias")] = v
        else:
            bayes_params[k] = v

    common_keys = sorted(set(dp_params) & set(bayes_params))
    print(f"Matched {len(common_keys)} parameters")

    report = {}
    for k in common_keys:
        w0 = dp_params[k].flatten()
        mu = bayes_params[k].flatten()

        diff = mu - w0
        report[k] = {
            "l2": diff.norm().item(),
            "rel_l2": (diff.norm() / (w0.norm() + 1e-12)).item(),
            "cos": (torch.dot(mu, w0) / (mu.norm() * w0.norm() + 1e-12)).item(),
            "shape": tuple(w0.shape)
        }
    return report

@click.command()
@click.option('-dc', '--dp_checkpoint', required=True)
@click.option('-bc', '--bayes_checkpoint', required=True)
@click.option('-o', '--output_dir', required=True)
def main(dp_checkpoint, bayes_checkpoint, output_dir):

    if os.path.exists(output_dir):
        click.confirm(f"Output path {output_dir} already exists! Overwrite?", abort=True)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # load Bayesian Diffusion Policy checkpoint
    payload_bayes = torch.load(open(bayes_checkpoint, 'rb'), pickle_module=dill)
    cfg_bayes = payload_bayes['cfg']
    cls = hydra.utils.get_class(cfg_bayes._target_)
    workspace = cls(cfg_bayes, output_dir=output_dir)
    workspace: BaseWorkspace
    workspace.load_payload(payload_bayes, exclude_keys=['model', 'optimizer'], include_keys=None)
    Bayes_model = workspace.model.model
    if cfg_bayes.training.use_ema:
        Bayes_model = workspace.ema_model.model

    # load NonBayesian Diffusion Policy checkpoint
    payload_dp = torch.load(open(dp_checkpoint, 'rb'), pickle_module=dill)
    cfg_dp = payload_dp['cfg']
    cls = hydra.utils.get_class(cfg_dp._target_)
    workspace = cls(cfg_dp, output_dir=output_dir)
    workspace: BaseWorkspace
    workspace.load_payload(payload_dp, exclude_keys=None, include_keys=None)
    DP_model = workspace.model.model
    if cfg_dp.training.use_ema:
        DP_model = workspace.ema_model.model

    # Compare models
    report1 = compare_mean_to_baseline(DP_model, Bayes_model)
    report2 = bayesian_uncertainty_report(DP_model, Bayes_model)

    out_path1 = os.path.join(output_dir, 'mu_analysis.json')
    out_path2 = os.path.join(output_dir, 'bayes_uncertainty_analysis.json')
    json.dump(report1, open(out_path1, 'w'), indent=2, sort_keys=True)
    json.dump(report2, open(out_path2, 'w'), indent=2, sort_keys=True)

if __name__ == '__main__':
    main()