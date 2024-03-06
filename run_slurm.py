import getpass
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Fake Colorama!
Fore = lambda x: x  # noqa: E731
Fore.CYAN = "\x1b[36m"
Fore.RESET = "\x1b[39m"

JOB_DIRECTORY = Path.home() / "slurm_jobs"


def cyan(x: str) -> str:
    return f"{Fore.CYAN}{x}{Fore.RESET}"


def submit_slurm_job(
    job_name: str,
    job_dir: Path,
    script: str,
    email: str,
    memory: str,
    num_gpus: int,
    num_cpus: int,
    time: str,
    partition: str,
    env_name: Optional[str] = None,
):
    slurm_file = f"""#!/bin/bash
#SBATCH -J {job_name}
#SBATCH -o {job_dir}/out_%j.out
#SBATCH -e {job_dir}/error_%j.err
#SBATCH --mail-user={email}
#SBATCH --mail-type=FAIL
#SBATCH --gres=gpu:{num_gpus}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task={num_cpus}
#SBATCH --mem={memory}
#SBATCH --partition=vision-sitzmann
#SBATCH --qos=vision-sitzmann-main
#SBATCH --time={time}

source {Path.home()}/.bashrc
source /scratch/charatan/miniconda3/etc/profile.d/conda.sh
cd {job_dir}
conda activate {env_name}

{script}
"""

    job_path = job_dir / "job.slurm"
    with job_path.open("w") as f:
        f.write(slurm_file)

    os.system(f"chmod +x {job_path}")
    os.system(f"sbatch {job_path}")

    print(f"{cyan('script:')} {script}")


if __name__ == "__main__":
    # Make sure we're in the right place (maybe)
    assert (Path.cwd() / ".git").exists()

    # Figure out what to call the run.
    command = " ".join(sys.argv[1:])
    name = None
    for name_source in ("-n", "--name"):
        try:
            index = sys.argv[1:].index(name_source)
            name = sys.argv[1:][index + 1]
            break
        except ValueError:
            pass

    # Create a directory for the run.
    day = datetime.now().strftime("%Y-%m-%d")
    time = datetime.now().strftime("%H-%M-%S")
    job_name = f"{time}_{name}"
    job_dir = JOB_DIRECTORY / "gaussian_barf" / day / job_name
    job_dir.mkdir(exist_ok=True, parents=True)

    # Copy/link the stuff that's needed to make the model run.
    for d in ("arguments", "gaussian_renderer", "lpipsPyTorch", "scene", "utils"):
        os.system(f"cp -r {d} {job_dir}/{d}")
    for d in ("datasets", "metrics", "assets", "psnrs"):
        os.system(f"ln -s {Path.cwd()}/{d} {job_dir}/{d}")
    for f in ("render.py", "run_slurm.py", "train.py", "metrics.py", "convert.py"):
        os.system(f"cp {f} {job_dir}/{f}")

    submit_slurm_job(
        job_name,
        job_dir,
        f"{command}",
        f"{getpass.getuser()}@mit.edu",
        "64G",
        1,
        8,
        "24:00:00",
        "",
        "/scratch/charatan/miniconda3/envs/flowmap",
    )
