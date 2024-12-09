import numpy as np
import skimage as sk

class HOG:

    def compute_gradient(self, img: np.ndarray, grad_filter: np.ndarray) -> np.ndarray:
        ts = grad_filter.shape[0]
        new_img = np.zeros((img.shape[0] + ts - 1, img.shape[1] + ts - 1))

        new_img[int((ts-1)/2.0):img.shape[0] + int((ts-1)/2.0), 
                int((ts-1)/2.0):img.shape[1] + int((ts-1)/2.0)] = img
        
        result = np.zeros((new_img.shape))
        
        for r in np.uint16(np.arange((ts-1)/2.0, img.shape[0]+(ts-1)/2.0)):
            for c in np.uint16(np.arange((ts-1)/2.0, img.shape[1]+(ts-1)/2.0)):
                curr_region = new_img[r-np.uint16((ts-1)/2.0):r+np.uint16((ts-1)/2.0)+1, 
                                    c-np.uint16((ts-1)/2.0):c+np.uint16((ts-1)/2.0)+1]
                curr_result = curr_region * grad_filter
                score = np.sum(curr_result)
                result[r, c] = score

        result_img = result[np.uint16((ts-1)/2.0):result.shape[0]-np.uint16((ts-1)/2.0), 
                            np.uint16((ts-1)/2.0):result.shape[1]-np.uint16((ts-1)/2.0)]

        return result_img

    def compute_gradient_magnitude(self, horizontal_gradient: np.ndarray, vertical_gradient: np.ndarray) -> np.ndarray:
        horizontal_squared = np.power(horizontal_gradient, 2)
        vertical_squared = np.power(vertical_gradient, 2)
        sum_of_squares = horizontal_squared + vertical_squared
        gradient_magnitude = np.sqrt(sum_of_squares)
        return gradient_magnitude

    def compute_gradient_direction(self, horizontal_gradient: np.ndarray, vertical_gradient: np.ndarray) -> np.ndarray:
        horizontal_gradient = horizontal_gradient + 1e-5
        gradient_direction_radians = np.arctan(vertical_gradient / horizontal_gradient)
        gradient_direction_degrees = np.rad2deg(gradient_direction_radians)
        gradient_direction_degrees = np.mod(gradient_direction_degrees, 180)
        return gradient_direction_degrees

    def find_nearest_bins(self, curr_direction: float, hist_bins: np.ndarray) -> (int, int):
        diff = np.abs(hist_bins - curr_direction)
        first_bin_idx = np.argmin(diff)
        
        if curr_direction < hist_bins[first_bin_idx]:
            second_bin_idx = (first_bin_idx - 1) % len(hist_bins)
        else:
            second_bin_idx = (first_bin_idx + 1) % len(hist_bins)
        
        return first_bin_idx, second_bin_idx

    def update_histogram_bins(
            self, 
            HOG_cell_hist: np.ndarray, 
            curr_direction: float, 
            curr_magnitude: float, 
            first_bin_idx: int, 
            second_bin_idx: int, 
            hist_bins: np.ndarray
        ) -> None:

        bin_size =  hist_bins[1] - hist_bins[0]
        first_bin_center = hist_bins[first_bin_idx]
        dist_to_first_bin = np.abs(curr_direction - first_bin_center)
        dist_to_second_bin = bin_size - dist_to_first_bin
        
        first_bin_contribution = curr_magnitude * (1 - dist_to_first_bin / bin_size)
        second_bin_contribution = curr_magnitude * (1 - dist_to_second_bin / bin_size)
        
        HOG_cell_hist[first_bin_idx] += first_bin_contribution
        HOG_cell_hist[second_bin_idx] += second_bin_contribution

    def calculate_histogram_per_cell(self, cell_direction: np.ndarray, cell_magnitude: np.ndarray, hist_bins: np.ndarray) -> np.ndarray:
        HOG_cell_hist = np.zeros(len(hist_bins))
        
        for r in range(cell_direction.shape[0]):
            for c in range(cell_direction.shape[1]):
                curr_direction = cell_direction[r, c]
                curr_magnitude = cell_magnitude[r, c]
                first_bin_idx, second_bin_idx = self.find_nearest_bins(curr_direction, hist_bins)
                self.update_histogram_bins(HOG_cell_hist, curr_direction, curr_magnitude, first_bin_idx, second_bin_idx, hist_bins)
        
        return HOG_cell_hist

    def compute_hog_features(self, image: np.ndarray) -> np.ndarray:
        # Define gradient masks
        horizontal_mask = np.array([-1, 0, 1])
        vertical_mask = np.array([[-1], [0], [1]])

        # Compute gradients
        horizontal_gradient = self.compute_gradient(image, horizontal_mask)
        vertical_gradient = self.compute_gradient(image, vertical_mask)

        # Compute gradient magnitude and direction
        grad_magnitude = self.compute_gradient_magnitude(horizontal_gradient, vertical_gradient)
        grad_direction = self.compute_gradient_direction(horizontal_gradient, vertical_gradient)

        # Define histogram bins
        hist_bins = np.array([10, 30, 50, 70, 90, 110, 130, 150, 170])

        # Compute histograms for each cell
        cells_histogram = np.zeros((16, 8, 9))
        for r in range(0, grad_magnitude.shape[0], 8):
            for c in range(0, grad_magnitude.shape[1], 8):
                cell_direction = grad_direction[r:r+8, c:c+8]
                cell_magnitude = grad_magnitude[r:r+8, c:c+8]
                cells_histogram[int(r / 8), int(c / 8)] = self.calculate_histogram_per_cell(cell_direction, cell_magnitude, hist_bins)

        # Normalize and concatenate histograms
        features_list = []
        for r in range(cells_histogram.shape[0] - 1):
            for c in range(cells_histogram.shape[1] - 1):
                histogram_16x16 = np.reshape(cells_histogram[r:r+2, c:c+2], (36,))
                histogram_16x16_normalized = histogram_16x16 / (np.linalg.norm(histogram_16x16) + 1e-5)
                features_list.append(histogram_16x16_normalized)

        return np.concatenate(features_list, axis=0)
