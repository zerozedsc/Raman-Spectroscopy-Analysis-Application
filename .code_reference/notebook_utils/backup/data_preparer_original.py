from sklearn.model_selection import train_test_split
from typing import List, Dict, Tuple, Any, Optional
import pickle
from scipy.interpolate import interp1d
import numpy as np  # Ensure numpy is imported

class RamanDataPreparer:
    """
    A class for preparing Raman spectroscopy data for machine learning tasks.
    
    This class handles data splitting at the patient level, unified wavelength interpolation,
    and preparation of new data for prediction. It ensures all spectra have consistent
    wavelength points and prevents data leakage in ML models.
    """
    
    def __init__(self, raman_data: Dict[str, Dict[str, List[Dict]]], 
                 selected_types: List[str] = None, 
                 test_size: float = 0.2, 
                 random_state: int = 42,
                 label_key: str = 'type',
                 batch_key: str = 'Hikkoshi',  # Key for batch labels (e.g., 'Hikkoshi')
                 wavelength_range: Optional[Tuple[int, int]] = None,
                 wavelength_step: int = 1,
                 apply_snv: bool = False):
        """
        Initialize the data preparer.
        
        Args:
            raman_data (dict): The full Raman data dictionary
            selected_types (list): List of data types to include (e.g., ['MGUS', 'MM', 'NL'])
            test_size (float): Proportion of patients to use for testing per batch (default 0.2). If 0, all data goes to training.
            random_state (int): Random seed for reproducible splits
            label_key (str): Key from metadata to use as labels ('type' or 'Hikkoshi')
            batch_key (str): Key from metadata to use as batch labels (default 'Hikkoshi')
            wavelength_range (tuple): Min and max wavelength for unified grid (default auto-detected)
            wavelength_step (int): Step size for wavelength grid (default 1 cm⁻¹)
            apply_snv (bool): Whether to apply Standard Normal Variate (SNV) preprocessing (default False)
        """
        self.raman_data = raman_data
        self.selected_types = selected_types or list(raman_data.keys())
        self.test_size = test_size
        self.random_state = random_state
        self.label_key = label_key
        self.batch_key = batch_key
        self.wavelength_range = wavelength_range
        self.wavelength_step = wavelength_step
        self.apply_snv = apply_snv
        
        # Validate selected types
        invalid_types = [t for t in self.selected_types if t not in self.raman_data]
        if invalid_types:
            raise ValueError(f"Invalid types selected: {invalid_types}")
        
        # Auto-detect wavelength range if not provided
        if self.wavelength_range is None:
            self.wavelength_range = self._auto_detect_wavelength_range()
        
        # Create unified wavelength grid
        self.unified_wavelengths = np.arange(self.wavelength_range[0], self.wavelength_range[1] + self.wavelength_step, self.wavelength_step)
        
        # Collect all patients
        self.all_patients = self._collect_all_patients()
        
        # Split patients with batch-aware stratification (or assign all to train if test_size=0)
        self.train_patients, self.test_patients = self._split_patients()
        
        print(f"Total patients: {len(self.all_patients)}")
        print(f"Train patients: {len(self.train_patients)}")
        print(f"Test patients: {len(self.test_patients)}")
        print(f"Unified wavelength grid: {len(self.unified_wavelengths)} points from {self.wavelength_range[0]} to {self.wavelength_range[1]} cm⁻¹")
        print(f"Using '{self.batch_key}' as batch key for batch-aware splitting.")
        print(f"SNV preprocessing: {'Enabled' if self.apply_snv else 'Disabled'}")
    
    def _auto_detect_wavelength_range(self) -> Tuple[int, int]:
        """
        Automatically detect the wavelength range from all spectra.
        
        Returns:
            Tuple of (min_wavelength, max_wavelength)
        """
        all_wavelengths = []
        for data_type in self.selected_types:
            if data_type in self.raman_data:
                for patient_id, spectra in self.raman_data[data_type].items():
                    for spectrum_data in spectra:
                        df = spectrum_data['dataframe']
                        wavelengths = df['wavelength'].values
                        all_wavelengths.extend(wavelengths)
        
        if not all_wavelengths:
            raise ValueError("No wavelength data found in the provided Raman data.")
        
        min_wavelength = int(np.floor(np.min(all_wavelengths)))
        max_wavelength = int(np.ceil(np.max(all_wavelengths)))
        
        print(f"Auto-detected wavelength range: {min_wavelength} to {max_wavelength} cm⁻¹")
        return (min_wavelength, max_wavelength)
    
    def _collect_all_patients(self) -> List[Tuple[str, str, str]]:
        """
        Collect all unique patients with their batch information.
        
        Returns:
            List of tuples: (data_type, patient_id, batch)
        """
        patients = []
        for data_type in self.selected_types:
            if data_type in self.raman_data:
                for patient_id in self.raman_data[data_type].keys():
                    # Determine batch from the first spectrum (assuming homogeneity per patient)
                    if self.raman_data[data_type][patient_id]:
                        batch = self.raman_data[data_type][patient_id][0]['metadata'].get(self.batch_key, 'Unknown')
                        patients.append((data_type, patient_id, batch))
        return patients
    
    def _split_patients(self) -> Tuple[List[Tuple[str, str, str]], List[Tuple[str, str, str]]]:
        """
        Split patients into train and test sets with batch-aware stratification.
        If test_size=0, all patients go to training.
        
        Groups patients by batch, then splits within each batch.
        
        Returns:
            Tuple of (train_patients, test_patients)
        """
        if self.test_size == 0:
            # No test split: all data for training
            print("test_size=0: All data allocated to training (no test split).")
            return self.all_patients, []
        
        # Group patients by batch
        batch_groups = {}
        for patient in self.all_patients:
            batch = patient[2]  # batch is the 3rd element
            if batch not in batch_groups:
                batch_groups[batch] = []
            batch_groups[batch].append(patient)
        
        train_patients = []
        test_patients = []
        
        for batch, patients in batch_groups.items():
            if len(patients) == 0:
                continue
            
            # Split patients within this batch
            batch_train, batch_test = train_test_split(
                patients, 
                test_size=self.test_size, 
                random_state=self.random_state,
                stratify=[p[0] for p in patients]  # Stratify by data type within batch
            )
            train_patients.extend(batch_train)
            test_patients.extend(batch_test)
            
            print(f"Batch '{batch}': {len(patients)} patients -> Train: {len(batch_train)}, Test: {len(batch_test)}")
        
        return train_patients, test_patients
  
    def _extract_spectra_from_patients(self, patients: List[Tuple[str, str, str]]) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
        """
        Extract spectra data and labels from a list of patients, with unified wavelength interpolation.
        
        Args:
            patients: List of (data_type, patient_id, batch) tuples
                
        Returns:
            Tuple of (X, y, sample_info, batch_labels)
        """
        spectra_list = []
        labels_list = []
        sample_info = []
        batch_labels = []
        
        for data_type, patient_id, _ in patients:  # Updated unpacking to handle 3-tuple; ignore batch with _
            if data_type in self.raman_data and patient_id in self.raman_data[data_type]:
                for spectrum_idx, spectrum_data in enumerate(self.raman_data[data_type][patient_id]):
                    df = spectrum_data['dataframe']
                    metadata = spectrum_data['metadata']
                    
                    # Ensure sorted by wavelength
                    df_sorted = df.sort_values('wavelength')
                    wavelengths = df_sorted['wavelength'].values
                    intensities = df_sorted['intensity'].values
                    
                    # Interpolate to unified wavelength grid
                    if len(wavelengths) > 1:
                        interp_func = interp1d(wavelengths, intensities, kind='linear', 
                                            bounds_error=False, fill_value=np.nan)
                        unified_intensities = interp_func(self.unified_wavelengths)
                        
                        # Handle any NaN values (outside original range)
                        if np.any(np.isnan(unified_intensities)):
                            # Use nearest valid values or zero-fill
                            valid_mask = ~np.isnan(unified_intensities)
                            if np.any(valid_mask):
                                unified_intensities = np.interp(self.unified_wavelengths, 
                                                            self.unified_wavelengths[valid_mask], 
                                                            unified_intensities[valid_mask])
                            else:
                                unified_intensities = np.zeros_like(self.unified_wavelengths)
                    else:
                        # If only one point, fill with that value
                        unified_intensities = np.full(len(self.unified_wavelengths), intensities[0] if len(intensities) > 0 else 0)
                    
                    spectra_list.append(unified_intensities)
                    labels_list.append(metadata.get(self.label_key, 'Unknown'))
                    sample_info.append(f"{data_type}_{patient_id}_spectrum_{spectrum_idx}")
                    batch_labels.append(metadata.get(self.batch_key, 'Unknown'))  # Use configurable batch key
        
        X = np.array(spectra_list)
        y = np.array(labels_list)
        
        return X, y, sample_info, batch_labels 
    
    def prepare_data(self) -> Dict[str, Any]:
        """
        Prepare the train/test data splits with unified wavelength interpolation.
        
        Returns:
            Dictionary containing:
            - X_train, X_test: Feature matrices (unified wavelengths)
            - y_train, y_test: Labels
            - train_patients, test_patients: Patient lists (with batch info)
            - train_info, test_info: Sample information
            - batch_train, batch_test: Batch labels
            - unified_wavelengths: The common wavelength grid
        """
        print("Preparing training data...")
        X_train, y_train, train_info, batch_train = self._extract_spectra_from_patients(self.train_patients)
        
        if len(self.test_patients) == 0:
            print("No test data (test_size=0): Skipping test data preparation.")
            X_test = np.array([])
            y_test = np.array([])
            test_info = []
            batch_test = []
        else:
            print("Preparing test data...")
            X_test, y_test, test_info, batch_test = self._extract_spectra_from_patients(self.test_patients)
        
        # Apply SNV if enabled
        if self.apply_snv:
            print("Applying SNV preprocessing to training data...")
            X_train = self._apply_snv(X_train)
            if len(X_test) > 0:
                print("Applying SNV preprocessing to test data...")
                X_test = self._apply_snv(X_test)
        
        print(f"Training set: {X_train.shape[0]} spectra, {X_train.shape[1]} wavelength points")
        if len(X_test) > 0:
            print(f"Test set: {X_test.shape[0]} spectra, {X_test.shape[1]} wavelength points")
        
        # Print label distribution
        unique_train_labels, train_counts = np.unique(y_train, return_counts=True)
        unique_test_labels, test_counts = np.unique(y_test, return_counts=True)
        
        print("\nTraining set label distribution:")
        for label, count in zip(unique_train_labels, train_counts):
            print(f"  {label}: {count}")
        
        if len(y_test) > 0:
            print("\nTest set label distribution:")
            for label, count in zip(unique_test_labels, test_counts):
                print(f"  {label}: {count}")
        
        # Print batch distribution
        unique_train_batches, train_batch_counts = np.unique(batch_train, return_counts=True)
        unique_test_batches, test_batch_counts = np.unique(batch_test, return_counts=True)
        
        print(f"\nTraining set batch distribution ({self.batch_key}):")
        for batch, count in zip(unique_train_batches, train_batch_counts):
            print(f"  {batch}: {count}")
        
        if len(batch_test) > 0:
            print(f"\nTest set batch distribution ({self.batch_key}):")
            for batch, count in zip(unique_test_batches, test_batch_counts):
                print(f"  {batch}: {count}")
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'train_patients': self.train_patients,
            'test_patients': self.test_patients,
            'train_info': train_info,
            'test_info': test_info,
            'batch_train': batch_train,
            'batch_test': batch_test,
            'unified_wavelengths': self.unified_wavelengths,
            'n_features': X_train.shape[1]
        }    
    
    def prepare_prediction_data_with_labels(self, new_raman_data: List[Tuple[Dict, str]]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare new Raman spectra data for prediction using the unified wavelength grid,
        and use the provided labels for compatibility with the trained model.
        
        Args:
            new_raman_data: List of tuples, each containing (spectrum_data_dict, desired_label_str)
                            where spectrum_data_dict has 'dataframe' and 'metadata' keys
                    
        Returns:
            Tuple[np.ndarray, np.ndarray, List[str]]: Prepared feature matrix, labels, and batch labels
        """
        spectra_list = []
        labels_list = []
        batch_labels = []
        
        for spectrum_data, desired_label in new_raman_data:  # Unpacking here
            df = spectrum_data['dataframe']
            metadata = spectrum_data['metadata']
            
            # Ensure sorted by wavelength
            df_sorted = df.sort_values('wavelength')
            wavelengths = df_sorted['wavelength'].values
            intensities = df_sorted['intensity'].values
            
            # Interpolate to unified wavelength grid
            if len(wavelengths) > 1:
                interp_func = interp1d(wavelengths, intensities, kind='linear', 
                                    bounds_error=False, fill_value=np.nan)
                unified_intensities = interp_func(self.unified_wavelengths)
                
                # Handle any NaN values (outside original range)
                if np.any(np.isnan(unified_intensities)):
                    # Use nearest valid values or zero-fill
                    valid_mask = ~np.isnan(unified_intensities)
                    if np.any(valid_mask):
                        unified_intensities = np.interp(self.unified_wavelengths, 
                                                    self.unified_wavelengths[valid_mask], 
                                                    unified_intensities[valid_mask])
                    else:
                        unified_intensities = np.zeros_like(self.unified_wavelengths)
            else:
                # If only one point, fill with that value
                unified_intensities = np.full(len(self.unified_wavelengths), intensities[0] if len(intensities) > 0 else 0)
            
            spectra_list.append(unified_intensities)
            labels_list.append(desired_label)
            batch_labels.append(metadata.get(self.batch_key, 'Unknown'))  # Use configurable batch key
        
        X_new = np.array(spectra_list)
        y_new = np.array(labels_list)
        
        # Apply SNV if enabled
        if self.apply_snv:
            print("Applying SNV preprocessing to new prediction data...")
            X_new = self._apply_snv(X_new)
        
        print(f"Prepared {X_new.shape[0]} new spectra for prediction with labels")
        print(f"Label distribution: {dict(zip(*np.unique(y_new, return_counts=True)))}")
        print(f"Batch distribution ({self.batch_key}): {dict(zip(*np.unique(batch_labels, return_counts=True)))}")
        
        return X_new, y_new, batch_labels
    
    def _apply_snv(self, X: np.ndarray) -> np.ndarray:
        """
        Applies Standard Normal Variate (SNV) to spectra.
        
        Args:
            X (np.ndarray): Feature matrix (spectra as rows)
            
        Returns:
            np.ndarray: SNV-transformed feature matrix
        """
        # Calculate mean and std dev for each spectrum (row)
        mean = np.mean(X, axis=1, keepdims=True)
        std_dev = np.std(X, axis=1, keepdims=True)
        
        # Apply SNV transformation
        X_snv = (X - mean) / std_dev
        return X_snv
    
    def save_split(self, filepath: str, data_dict: Dict[str, Any]):
        """
        Save the prepared data split to a pickle file.
        
        Args:
            filepath (str): Path to save the file
            data_dict (dict): Data dictionary from prepare_data()
        """
        with open(filepath, 'wb') as f:
            pickle.dump(data_dict, f)
        print(f"Data split saved to {filepath}")
    
    def get_patient_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about patients and spectra distribution.
        
        Returns:
            Dictionary with statistics
        """
        stats = {
            'total_patients': len(self.all_patients),
            'train_patients': len(self.train_patients),
            'test_patients': len(self.test_patients),
            'patients_per_type': {}
        }
        
        for data_type in self.selected_types:
            type_patients = [p for p in self.all_patients if p[0] == data_type]
            stats['patients_per_type'][data_type] = len(type_patients)
            
            # Count spectra per type
            total_spectra = 0
            for _, patient_id in type_patients:
                if patient_id in self.raman_data[data_type]:
                    total_spectra += len(self.raman_data[data_type][patient_id])
            stats['patients_per_type'][data_type] = {
                'patients': len(type_patients),
                'spectra': total_spectra
            }
        
        return stats

    # New visualization methods
    def plot_pca(self, type_key: str = "MGUS", group_by: str = "Hikkoshi", 
                 n_components: int = 2, title: str = None, show_figure: bool = True, **kwargs) -> dict:
        """
        Plot PCA for a given type, colored by group_by metadata.
        
        Args:
            type_key (str): Data type to plot (e.g., 'MGUS')
            group_by (str): Metadata key for coloring (e.g., 'Hikkoshi')
            n_components (int): Number of PCA components
            title (str): Custom title for the plot
            show_figure (bool): Whether to display the plot
            **kwargs: Additional arguments for PCA
        
        Returns:
            dict: Result from RamanSpectrumAdvanceVisualize.plot_raman_pca
        """
        visualizer = RamanSpectrumAdvanceVisualize(self.raman_data)
        return visualizer.plot_raman_pca(
            type_key=type_key, group_by=group_by, n_components=n_components, 
            title=title, show_figure=show_figure, **kwargs
        )
    
    def plot_umap(self, type_keys: List[str], group_by: str = "Hikkoshi", 
                  n_neighbors: int = 15, min_dist: float = 0.1, metric: str = "euclidean", 
                  title: str = None, random_state: int = 42, **kwargs) -> dict:
        """
        Plot UMAP for selected type_keys, colored by type and shaped by group_by metadata.
        
        Args:
            type_keys (List[str]): List of data types to plot (e.g., ['MGUS', 'MM'])
            group_by (str): Metadata key for shaping markers (e.g., 'Hikkoshi')
            n_neighbors (int): Number of neighbors for UMAP
            min_dist (float): Minimum distance for UMAP
            metric (str): Distance metric for UMAP
            title (str): Custom title for the plot
            random_state (int): Random state for reproducibility
            **kwargs: Additional arguments for UMAP
        
        Returns:
            dict: Result from RamanSpectrumAdvanceVisualize.plot_raman_umap
        """
        visualizer = RamanSpectrumAdvanceVisualize(self.raman_data)
        return visualizer.plot_raman_umap(
            type_keys=type_keys, group_by=group_by, n_neighbors=n_neighbors, 
            min_dist=min_dist, metric=metric, title=title, random_state=random_state, **kwargs
        )
    
