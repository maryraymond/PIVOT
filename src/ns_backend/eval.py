# PIVOT: Pose, Intrinsics and Viewpoint Oriented Testbed
# Copyright (c) 2026 Mary Raymond
# Date: 2026-07-07
# MIT License — see LICENSE in the project root for details.

# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/usr/bin/env python
"""
eval.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import tyro

from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.utils.rich_utils import CONSOLE

from nerfstudio.scripts.nerfstudio_drone_eval import calculate_drone_ds_metrcis
from nerfstudio.scripts.ns_models_trajectory_eval import NsModel


@dataclass
class ComputePSNR:
    """Load a checkpoint, compute some PSNR metrics, and save it to a JSON file."""

    # Path to config YAML file.
    load_config: Path
    # Name of the output file.
    output_path: Path = Path("output.json")
    # Optional path to save rendered outputs to.
    render_output_path: Optional[Path] = None
    # temp solution for running drone ds eval
    run_drone_ds_eval:bool = False
    # if the evaluation datset if different from the one used for training
    # The root path to the evaluation dataset should be defined here
    eval_dataset: Optional[str] = None

    def main(self) -> None:

        if self.run_drone_ds_eval:
            # Create the Model object
            CONSOLE.print(f"Running drone DS eval")

            model = NsModel(self.load_config, 
                            compose_background=False)
            calculate_drone_ds_metrcis(model=model, 
                                       out_dir=self.output_path, 
                                       eval_dataset=self.eval_dataset,
                                       debug=False, 
                                       normalization_scale="scene_diameter")
            
        else:
            """Main function."""
            config, pipeline, checkpoint_path, _ = eval_setup(self.load_config)
            assert self.output_path.suffix == ".json"
            if self.render_output_path is not None:
                self.render_output_path.mkdir(parents=True, exist_ok=True)
            metrics_dict = pipeline.get_average_eval_image_metrics(output_path=self.render_output_path, get_std=True)
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            # Get the output and define the names to save to
            benchmark_info = {
                "experiment_name": config.experiment_name,
                "method_name": config.method_name,
                "checkpoint": str(checkpoint_path),
                "results": metrics_dict,
            }
            # Save output to output file
            self.output_path.write_text(json.dumps(benchmark_info, indent=2), "utf8")
            CONSOLE.print(f"Saved results to: {self.output_path}")


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(ComputePSNR).main()


if __name__ == "__main__":
    entrypoint()

# For sphinx docs
get_parser_fn = lambda: tyro.extras.get_parser(ComputePSNR)  # noqa
