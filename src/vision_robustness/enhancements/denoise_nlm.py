"""Non-Local Means denoise (course enhancement for additive noise).

Uses OpenCV ``fastNlMeansDenoisingColored`` with noise-adaptive filter strength
``h ∝ σ``. This matches the lecture choice (NLM for additive noise) while staying
practical for large evaluation runs. scikit-image NLM is available as a
higher-quality optional backend.
"""

from __future__ import annotations

import numpy as np

from vision_robustness.enhancements.base import Enhancement, EnhancementResult


class NonLocalMeans(Enhancement):
    name = "non_local_means"

    def __init__(
        self,
        h_factor: float = 0.85,
        template_window: int = 7,
        search_window: int = 21,
        backend: str = "opencv",  # opencv | skimage
        bilateral_refine: bool = False,
    ):
        self.h_factor = float(h_factor)
        self.template_window = int(template_window)
        self.search_window = int(search_window)
        self.backend = backend
        self.bilateral_refine = bool(bilateral_refine)

    def apply(self, image: np.ndarray, **kwargs) -> EnhancementResult:
        noise_std = kwargs.get("noise_std")
        if noise_std is not None and float(noise_std) > 0:
            sigma_px = float(noise_std)
            sigma_source = "provided"
        else:
            sigma_px = self._estimate_sigma_px(image)
            sigma_source = "estimated"

        # OpenCV h is roughly on the pixel-intensity scale
        h = float(np.clip(self.h_factor * sigma_px, 3.0, 45.0))

        if self.backend == "skimage":
            out = self._skimage_nlm(image, sigma_px / 255.0)
            backend = "skimage"
        else:
            out = self._opencv_nlm(image, h)
            backend = "opencv"

        if self.bilateral_refine:
            import cv2

            out = cv2.bilateralFilter(out, d=5, sigmaColor=40, sigmaSpace=40)

        return EnhancementResult(
            image=out,
            name=self.name,
            params={
                "backend": backend,
                "sigma_px": sigma_px,
                "sigma_source": sigma_source,
                "h": h,
                "h_factor": self.h_factor,
                "template_window": self.template_window,
                "search_window": self.search_window,
            },
        )

    @staticmethod
    def _estimate_sigma_px(image_u8: np.ndarray) -> float:
        from skimage.restoration import estimate_sigma

        img01 = image_u8.astype(np.float64) / 255.0
        return float(estimate_sigma(img01, channel_axis=-1, average_sigmas=True) * 255.0)

    def _opencv_nlm(self, image_u8: np.ndarray, h: float) -> np.ndarray:
        import cv2

        bgr = cv2.cvtColor(image_u8, cv2.COLOR_RGB2BGR)
        denoised = cv2.fastNlMeansDenoisingColored(
            bgr,
            None,
            h=h,
            hColor=h,
            templateWindowSize=self.template_window,
            searchWindowSize=self.search_window,
        )
        return cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB)

    def _skimage_nlm(self, image_u8: np.ndarray, sigma01: float) -> np.ndarray:
        from skimage.restoration import denoise_nl_means

        img = image_u8.astype(np.float64) / 255.0
        h = self.h_factor * max(sigma01, 1.0 / 255.0)
        out = denoise_nl_means(
            img,
            h=h,
            patch_size=5,
            patch_distance=6,
            channel_axis=-1,
            fast_mode=True,
        )
        return np.clip(np.round(out * 255.0), 0, 255).astype(np.uint8)
