(function () {
  const html = document.documentElement;
  const themeToggle = document.getElementById('themeToggle');
  const predictForm = document.getElementById('predictForm');
  const loaderOverlay = document.getElementById('loaderOverlay');
  const imageInput = document.getElementById('imageInput');
  const imagePreview = document.getElementById('imagePreview');

  const storedTheme = localStorage.getItem('deepfake-theme') || 'light';
  html.setAttribute('data-theme', storedTheme);

  if (themeToggle) {
    themeToggle.addEventListener('click', function () {
      const nextTheme = html.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
      html.setAttribute('data-theme', nextTheme);
      localStorage.setItem('deepfake-theme', nextTheme);
    });
  }

  if (imageInput && imagePreview) {
    imageInput.addEventListener('change', function () {
      const file = this.files && this.files[0];
      if (!file) {
        return;
      }
      const reader = new FileReader();
      reader.onload = function (event) {
        imagePreview.src = event.target.result;
      };
      reader.readAsDataURL(file);
    });
  }

  if (predictForm && loaderOverlay) {
    predictForm.addEventListener('submit', function () {
      loaderOverlay.classList.add('visible');
    });
  }
})();
