from huggingface_hub import hf_hub_download
from huggingface_hub import snapshot_download

ckpt = hf_hub_download(
    repo_id="Respair/Tsukasa_Speech",
    filename="Models/Style_Tsukasa_v02/Top_ckpt_24khz.pth",
    local_dir=".",
)

ckpt = hf_hub_download(
    repo_id="Respair/Tsukasa_Speech",
    filename="Utils/ASR/bst_00080.pth",
    local_dir=".",
)

ckpt = hf_hub_download(
    repo_id="Respair/Tsukasa_Speech",
    filename="Utils/JDC/bst.t7",
    local_dir=".",
)

# Using snapshot_download to download a folder
snapshot_download(
    repo_id="Respair/Tsukasa_Speech",
    allow_patterns="Utils/KTD/prompt_enc/checkpoint-73285/**",
    local_dir=".",
)

snapshot_download(
    repo_id="Respair/Tsukasa_Speech",
    allow_patterns="Utils/PLBERT/**",
    local_dir=".",
)

snapshot_download(
    repo_id="Respair/Tsukasa_Speech",
    allow_patterns="Utils/KTD/text_enc/checkpoint-22680/**",
    local_dir="."
)