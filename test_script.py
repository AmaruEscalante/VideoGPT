import argparse
from videogpt.gpt import VideoGPT
from videogpt import download, load_vqvae, load_videogpt
from videogpt.vqvae import VQVAE
from videogpt.download import load_i3d_pretrained
from tqdm import tqdm
import numpy as np
import torch
from videogpt.fvd.fvd import get_fvd_logits, frechet_distance
from videogpt import VideoData, VideoGPT, load_videogpt

# Initialize argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--trials", type=int, default=1, help="Number of trials to run")
parser.add_argument(
    "--batch_size", type=int, default=16, help="Batch size for processing"
)
parser.add_argument(
    "--max_batch", type=int, default=32, help="Maximum batch size for processing"
)
args = parser.parse_args()

# Download VQ-VAE
device = torch.device("cuda")
filepath = download("1FNWJtWDTX5CcVSSlINK1ZFFHuBgjBZfB", "ucf101_stride4x4x4")
vqvae = VQVAE.load_from_checkpoint(filepath).to(device)
# Download Pre-trained GPT
filepath = download("1c4CYL1joN5KDC5VYJIilFYWcDOmjWtgE", "ucf101_uncond_gpt")
gpt = VideoGPT.load_from_checkpoint(filepath).to(device)


def main(ckpt="bair_gpt", n_trials=1, batch_size=16, max_batch=32):
    torch.set_grad_enabled(False)
    #################### Load VideoGPT ########################################
    gpt.eval()
    hparams = gpt.hparams["args"]
    # print("hparams", hparams)
    hparams.batch_size = batch_size
    loader = VideoData(hparams).test_dataloader()

    #################### Load I3D ########################################
    i3d = load_i3d_pretrained(device)

    #################### Compute FVD ###############################
    fvds = []
    fvds_star = []
    pbar = tqdm(total=n_trials)
    for _ in range(n_trials):
        fvd, fvd_star = eval_fvd(i3d, gpt, loader, device, max_batch)
        fvds.append(fvd)
        fvds_star.append(fvd_star)

        pbar.update(1)
        fvd_mean = np.mean(fvds)
        fvd_std = np.std(fvds)

        fvd_star_mean = np.mean(fvds_star)
        fvd_star_std = np.std(fvds_star)

        pbar.set_description(
            f"FVD {fvd_mean:.2f} +/- {fvd_std:.2f}, FVD* {fvd_star_mean:.2f} +/- {fvd_star_std:.2f}"
        )
    pbar.close()
    print(
        f"Final FVD {fvd_mean:.2f} +/- {fvd_std:.2f}, FVD* {fvd_star_mean:.2f} +/- {fvd_star_std:.2f}"
    )


def eval_fvd(i3d, videogpt, loader, device, max_batch):
    batch = next(iter(loader))
    batch = {k: v.to(device) for k, v in batch.items()}

    fake_embeddings = []
    for i in range(0, batch["video"].shape[0], max_batch):
        fake = videogpt.sample(
            max_batch, {k: v[i : i + max_batch] for k, v in batch.items()}
        )
        fake = torch.repeat_interleave(fake, 4, dim=2)  # TODO: check correctness
        fake = fake.permute(0, 2, 3, 4, 1).cpu().numpy()  # BCTHW -> BTHWC
        fake = (fake * 255).astype("uint8")
        fake_embeddings.append(get_fvd_logits(fake, i3d=i3d, device=device))
    fake_embeddings = torch.cat(fake_embeddings)

    real = batch["video"].to(device)
    real_recon_embeddings = []
    for i in range(0, batch["video"].shape[0], max_batch):
        real_recon = (
            videogpt.get_reconstruction(batch["video"][i : i + max_batch]) + 0.5
        ).clamp(0, 1)
        real_recon = torch.repeat_interleave(real_recon, 4, dim=2)
        real_recon = real_recon.permute(0, 2, 3, 4, 1).cpu().numpy()
        real_recon = (real_recon * 255).astype("uint8")
        real_recon_embeddings.append(get_fvd_logits(real_recon, i3d=i3d, device=device))
    real_recon_embeddings = torch.cat(real_recon_embeddings)

    real = real + 0.5
    real = real.permute(0, 2, 3, 4, 1).cpu().numpy()  # BCTHW -> BTHWC
    real = (real * 255).astype("uint8")
    real_embeddings = get_fvd_logits(real, i3d=i3d, device=device)
    # Ensure that fake_embeddings and real_embeddings have the same number of items
    assert (
        fake_embeddings.shape[0]
        == real_recon_embeddings.shape[0]
        == real_embeddings.shape[0]
    )

    fvd = frechet_distance(fake_embeddings.clone(), real_embeddings)
    fvd_star = frechet_distance(fake_embeddings.clone(), real_recon_embeddings)
    return fvd.item(), fvd_star.item()


if __name__ == "__main__":
    main(
        ckpt="ufc",
        n_trials=args.trials,
        batch_size=args.batch_size,
        max_batch=args.max_batch,
    )
