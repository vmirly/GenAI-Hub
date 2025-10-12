"""CogVLM single image inference script

This script provides a minimal example of how to load a CogVLM or CogAgent model
from Hugging Face and run it on a single image.  It is based on the
`basic_demo/cli_demo_hf.py` example provided in the CogVLM repository but
removes the interactive loop and instead accepts all inputs via command line
arguments.  The default model used is the general chat version of CogVLM, but
you can override the model to load any other checkpoint (for example one of
the grounding or agent variants) by supplying the `--model` flag.

Example usage (after installing dependencies and activating your conda
environment) to describe a plant image stored at ``../samples/plants.jpg``::

    python cogvlm_inference.py \
        --image ../samples/plants.jpg \
        --query "Please describe the image in detail." \
        --model THUDM/cogvlm-chat-hf \
        --bf16

If you wish to use 4‑bit quantization to reduce memory consumption, add the
``--quant 4`` option.  Quantization requires half precision (``--fp16``) rather
than bfloat16.  See the repository README for details on which versions of the
model support quantized inference.
"""

import argparse
from typing import Optional

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, LlamaTokenizer


def run_inference(
    model_name: str,
    image_path: str,
    query: str,
    *,
    quant: Optional[int] = None,
    use_fp16: bool = False,
    use_bf16: bool = False,
    tokenizer_name: str = "lmsys/vicuna-7b-v1.5",
) -> str:
    """Load a CogVLM/CogAgent model and perform a single image query.

    Parameters
    ----------
    model_name : str
        Hugging Face identifier of the model or path to a local checkpoint.
    image_path : str
        Path to the input image.  The image will be loaded using Pillow and
        converted to RGB.
    query : str
        The natural language query to ask about the image.  Examples include
        "Describe the image" or more targeted questions like
        "How many plants are there?".
    quant : Optional[int], default = ``None``
        If set to 4, load the model in 4‑bit quantized mode.  Quantized
        inference dramatically reduces GPU memory usage but may require a
        slower ``fp16`` data type instead of bfloat16.  Leave ``None`` to
        disable quantization.
    use_fp16 : bool, default = ``False``
        If ``True``, load the model using 16‑bit floating point.  This can
        reduce memory usage on GPUs that do not support bfloat16.  When
        ``quant`` is not ``None`` this option is implicitly enabled.
    use_bf16 : bool, default = ``False``
        If ``True``, load the model using the bfloat16 dtype.  Recommended for
        GPUs with bfloat16 support (e.g. NVIDIA Ampere and newer).  If both
        ``use_fp16`` and ``use_bf16`` are ``False``, the script defaults to
        half precision (FP16).
    tokenizer_name : str, default = ``"lmsys/vicuna-7b-v1.5"``
        Hugging Face identifier of the tokenizer to use.  The CogVLM family
        expects a Vicuna‑compatible tokenizer; if you run into problems
        downloading the tokenizer, pass ``--tokenizer_name /path/to/vicuna``.

    Returns
    -------
    str
        The model's response to the query.
    """

    # Determine computation device: use GPU if available, otherwise fall back to CPU
    device = "cpu" #"cuda" if torch.cuda.is_available() else "cpu"

    # Choose appropriate floating point type
    if use_bf16 and not use_fp16:
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float16

    # Load tokenizer.  The Vicuna tokenizer is required for CogVLM/CogAgent models.
    tokenizer = LlamaTokenizer.from_pretrained(tokenizer_name)

    # Load model.  When quantization is requested we call from_pretrained with
    # ``load_in_4bit=True`` and do not move the model to the device (the
    # quantized model handles device placement internally).  Otherwise, load the
    # model and explicitly move it onto ``device`` for faster inference.
    if quant:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            load_in_4bit=True,
            trust_remote_code=True,
        )
        model.eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        model.to(device)
        model.eval()

    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")

    # Build the conversation input.  CogVLM models provide a helper method
    # `build_conversation_input_ids` which assembles the token and image
    # tensors required for multimodal generation.  Here we start with an empty
    # history since this script performs a single turn.
    input_by_model = model.build_conversation_input_ids(
        tokenizer,
        query=query,
        history=[],
        images=[image],
    )

    # Construct the dict of tensors expected by the model.generate API.  Note
    # that ``images`` and ``cross_images`` tensors live on the CPU until
    # generation time.  They are moved to ``device`` and cast to the model's
    # dtype as needed.  For quantized models ``images`` must remain as a
    # nested list (one list per batch) of tensors.
    inputs = {
        "input_ids": input_by_model["input_ids"].unsqueeze(0).to(device),
        "token_type_ids": input_by_model["token_type_ids"].unsqueeze(0).to(device),
        "attention_mask": input_by_model["attention_mask"].unsqueeze(0).to(device),
        "images": [[input_by_model["images"][0].to(device).to(torch_dtype)]],
    }

    # Some models also return ``cross_images`` for cross‑attention layers
    if "cross_images" in input_by_model and input_by_model["cross_images"]:
        inputs["cross_images"] = [[input_by_model["cross_images"][0].to(device).to(torch_dtype)]]

    # Generation parameters: disable sampling for deterministic output and limit
    # the total sequence length.  Feel free to adjust ``max_length`` to suit
    # your needs.
    gen_kwargs = {
        "max_length": 2048,
        "do_sample": False,
    }

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
        # Slice off the input tokens to obtain only newly generated tokens
        new_tokens = outputs[:, inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(new_tokens[0], skip_special_tokens=True)

    return response


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CogVLM/CogAgent on a single image")
    parser.add_argument(
        "--model",
        default="THUDM/cogvlm-chat-hf",
        help="Hugging Face model name or local checkpoint path (default: THUDM/cogvlm-chat-hf)",
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Path to the image to analyse",
    )
    parser.add_argument(
        "--query",
        default="Please describe the image.",
        help="Text query to ask about the image",
    )
    parser.add_argument(
        "--quant",
        type=int,
        choices=[4],
        default=None,
        help="Enable 4‑bit quantized loading (reduces GPU memory usage)",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Load model weights in 16‑bit floating point (overrides bfloat16)",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Load model weights in bfloat16 (default when supported)",
    )
    parser.add_argument(
        "--tokenizer_name",
        default="lmsys/vicuna-7b-v1.5",
        help="Tokenizer name or path (default: lmsys/vicuna-7b-v1.5)",
    )
    args = parser.parse_args()

    # When quantization is enabled, force fp16: xformers/4‑bit quantization does not
    # support bfloat16.  If the user explicitly requested bf16 with quantization,
    # fall back to fp16 and warn.
    if args.quant and args.bf16:
        print("Warning: 4‑bit quantization is incompatible with bfloat16. "
              "Falling back to fp16.")
        args.bf16 = False
        args.fp16 = True

    response = run_inference(
        model_name=args.model,
        image_path=args.image,
        query=args.query,
        quant=args.quant,
        use_fp16=args.fp16,
        use_bf16=args.bf16,
        tokenizer_name=args.tokenizer_name,
    )
    print("Model response:\n{}".format(response))


if __name__ == "__main__":
    main()