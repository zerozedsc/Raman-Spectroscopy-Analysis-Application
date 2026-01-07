from .__utils__ import *
from .pipeline import PipelineStep


class PreprocessingThread(QThread):
    """Background thread for preprocessing operations with detailed progress updates."""

    progress_updated = Signal(int)
    status_updated = Signal(str)
    step_completed = Signal(str, int)  # step_name, step_number
    processing_completed = Signal(dict)
    processing_error = Signal(str)
    step_failed = Signal(str, str)  # step_name, error_message

    def __init__(
        self,
        pipeline_steps: List[PipelineStep],
        input_dfs: List[pd.DataFrame],
        output_name: str,
        parent=None,
    ):
        super().__init__(parent)
        self.pipeline_steps = pipeline_steps
        self.input_dfs = input_dfs
        self.output_name = output_name
        self.is_cancelled = False
        self.failed_steps = []

    def run(self):
        """Execute the preprocessing pipeline sequentially with error handling."""
        try:
            create_logs(
                "PreprocessingThread",
                "start",
                f"Starting preprocessing with {len(self.input_dfs)} datasets, output_name={self.output_name}",
                status="info",
            )

            self.status_updated.emit(LOCALIZE("PREPROCESS.STATUS.preparing_data"))
            self.progress_updated.emit(5)

            if self.is_cancelled:
                return

            # Merge input data first
            self.status_updated.emit(LOCALIZE("PREPROCESS.STATUS.merging_data"))
            self.progress_updated.emit(10)

            # Combine all DataFrames
            try:
                merged_df = pd.concat(self.input_dfs, axis=1)
                create_logs(
                    "PreprocessingThread",
                    "data_merge",
                    f"Merged data shape: {merged_df.shape}",
                    status="info",
                )
            except Exception as e:
                error_msg = f"Error merging input data: {str(e)}"
                create_logs(
                    "PreprocessingThread", "merge_error", error_msg, status="error"
                )
                self.processing_error.emit(error_msg)
                return

            # Initialize SpectralContainer with better error handling
            try:
                if merged_df.index.name == "wavenumber":
                    wavenumbers = merged_df.index.values
                    intensities = merged_df.values.T
                else:
                    if "wavenumber" in merged_df.columns:
                        wavenumbers = merged_df["wavenumber"].values
                        intensities = merged_df.drop("wavenumber", axis=1).values.T
                    else:
                        wavenumbers = merged_df.iloc[:, 0].values
                        intensities = merged_df.iloc[:, 1:].values.T

                create_logs(
                    "PreprocessingThread",
                    "data_structure",
                    f"Wavenumbers shape: {wavenumbers.shape}, Intensities shape: {intensities.shape}",
                    status="info",
                )

                spectra = rp.SpectralContainer(intensities, wavenumbers)

            except Exception as e:
                error_msg = f"Error creating SpectralContainer: {str(e)}"
                create_logs(
                    "PreprocessingThread", "container_error", error_msg, status="error"
                )
                self.processing_error.emit(error_msg)
                return

            self.progress_updated.emit(20)

            # Process each step sequentially
            total_steps = len(self.pipeline_steps)
            successful_steps = []
            skipped_steps = []

            create_logs(
                "PreprocessingThread",
                "pipeline_start",
                f"Processing {total_steps} pipeline steps",
                status="info",
            )

            for i, step in enumerate(self.pipeline_steps):
                if self.is_cancelled:
                    create_logs(
                        "PreprocessingThread",
                        "cancelled",
                        "Processing cancelled by user",
                        status="info",
                    )
                    return

                step_name = step.method
                step_progress_start = 20 + int((i / total_steps) * 60)
                step_progress_end = 20 + int(((i + 1) / total_steps) * 60)

                # Check if this step should be skipped (only skip if it's existing AND not enabled)
                if (
                    hasattr(step, "is_existing")
                    and step.is_existing
                    and not step.enabled
                ):
                    self.status_updated.emit(
                        LOCALIZE(
                            "PREPROCESS.STATUS.skipping_existing_step",
                            step=step_name,
                            number=i + 1,
                            total=total_steps,
                        )
                    )

                    skipped_steps.append(
                        {
                            "step_name": step_name,
                            "step_index": i + 1,
                            "category": step.category,
                            "reason": "existing_step_disabled",
                        }
                    )

                    self.progress_updated.emit(step_progress_end)
                    continue

                # Process the step (both new steps and enabled existing steps)
                self.status_updated.emit(
                    LOCALIZE(
                        "PREPROCESS.STATUS.processing_step",
                        step=step_name,
                        number=i + 1,
                        total=total_steps,
                    )
                )
                self.progress_updated.emit(step_progress_start)

                try:
                    create_logs(
                        "PreprocessingThread",
                        "step_start",
                        f"Starting step {i+1}/{total_steps}: {step_name} with params: {step.params}",
                        status="info",
                    )

                    # Create preprocessing instance
                    instance = step.create_instance()

                    # Apply preprocessing step
                    pre_shape = spectra.spectral_data.shape
                    pre_axis_shape = spectra.spectral_axis.shape

                    spectra = instance.apply(spectra)

                    post_shape = spectra.spectral_data.shape
                    post_axis_shape = spectra.spectral_axis.shape

                    # Log successful step
                    successful_steps.append(
                        {
                            "step_name": step_name,
                            "step_index": i + 1,
                            "category": step.category,
                            "parameters": step.params,
                            "data_change": {
                                "input_shape": pre_shape,
                                "output_shape": post_shape,
                                "axis_input_shape": pre_axis_shape,
                                "axis_output_shape": post_axis_shape,
                            },
                        }
                    )

                    self.step_completed.emit(step_name, i + 1)
                    self.progress_updated.emit(step_progress_end)

                    create_logs(
                        "PreprocessingThread",
                        "step_success",
                        f"Step {i+1}/{total_steps} ({step_name}) completed successfully. "
                        f"Data shape: {pre_shape} -> {post_shape}",
                        status="info",
                    )

                except Exception as e:
                    # Log failed step but continue processing
                    error_msg = f"Step {i+1} ({step_name}) failed: {str(e)}"
                    self.failed_steps.append(
                        {
                            "step_name": step_name,
                            "step_index": i + 1,
                            "category": step.category,
                            "error": str(e),
                            "error_type": type(e).__name__,
                        }
                    )

                    self.step_failed.emit(step_name, str(e))

                    create_logs(
                        "PreprocessingThread",
                        "step_error",
                        f"Step {i+1}/{total_steps} ({step_name}) failed: {e}. Continuing with remaining steps.",
                        status="error",
                    )

                    # Continue to next step
                    continue

            if self.is_cancelled:
                return

            # Finalize results
            self.progress_updated.emit(85)
            self.status_updated.emit(LOCALIZE("PREPROCESS.STATUS.finalizing_results"))

            # Create output DataFrame
            try:
                processed_df = pd.DataFrame(
                    spectra.spectral_data.T,
                    index=spectra.spectral_axis,
                    columns=[
                        f"{self.output_name}_{i}"
                        for i in range(spectra.spectral_data.shape[0])
                    ],
                )
                processed_df.index.name = "wavenumber"

                create_logs(
                    "PreprocessingThread",
                    "output_created",
                    f"Output DataFrame created with shape: {processed_df.shape}",
                    status="info",
                )

            except Exception as e:
                error_msg = f"Error creating output DataFrame: {str(e)}"
                create_logs(
                    "PreprocessingThread", "output_error", error_msg, status="error"
                )
                self.processing_error.emit(error_msg)
                return

            self.progress_updated.emit(100)
            self.status_updated.emit(LOCALIZE("PREPROCESS.STATUS.completed"))

            # Return comprehensive results
            result_data = {
                "processed_df": processed_df,
                "output_name": self.output_name,  # Include output name for proper saving
                "successful_steps": successful_steps,
                "failed_steps": self.failed_steps,
                "skipped_steps": skipped_steps,
                "total_steps": total_steps,
                "success_rate": (
                    len(successful_steps) / total_steps if total_steps > 0 else 0
                ),
                "spectra": spectra,
                "original_data": merged_df,
            }

            create_logs(
                "PreprocessingThread",
                "completed",
                f"Processing completed. Success rate: {result_data['success_rate']:.1%}",
                status="info",
            )

            create_logs(
                "PreprocessingThread",
                "emitting_completed",
                f"Emitting processing_completed signal for output_name={self.output_name}",
                status="info",
            )
            self.processing_completed.emit(result_data)

            create_logs(
                "PreprocessingThread",
                "completed_signal_emitted",
                "processing_completed signal emitted successfully",
                status="info",
            )

        except Exception as e:
            error_msg = f"{LOCALIZE('PREPROCESS.STATUS.error')}: {str(e)}"
            create_logs(
                "PreprocessingThread",
                "critical_error",
                f"Critical preprocessing error: {e}\n{traceback.format_exc()}",
                status="error",
            )
            self.processing_error.emit(error_msg)

        finally:
            create_logs(
                "PreprocessingThread",
                "run_finished",
                f"Thread run() method finished for output_name={self.output_name}",
                status="info",
            )

    def cancel(self):
        """Cancel the preprocessing operation."""
        self.is_cancelled = True
        create_logs(
            "PreprocessingThread",
            "cancel_requested",
            "Cancellation requested",
            status="info",
        )
        self.quit()
        self.wait()
