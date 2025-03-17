from .preprocess_threshold import preprocess_dataset
import numpy as np
def masking(args):
    if args.frequency_masking:
        threshold_dir = (args.root_dir / args.dataset.split(":")[0] / "thresholds") # If no threshold path, create it
        threshold_dir.mkdir(exist_ok=True,parents=True)

        mask_path = threshold_dir / f"{args.dataset}.np.npy"
        if not mask_path.exists():
            print(f"Threshold for dataset \'{args.dataset}\' not found. Calculating now:")
            threshold = preprocess_dataset(args.dataset,output=mask_path, batch_size=args.batch_size)
        else:
            threshold = np.load(mask_path)
    return threshold