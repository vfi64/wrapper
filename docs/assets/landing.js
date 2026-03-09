(function () {
  const revealItems = document.querySelectorAll(".reveal");
  const revealObserver = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add("is-visible");
          revealObserver.unobserve(entry.target);
        }
      });
    },
    { threshold: 0.16, rootMargin: "0px 0px -24px 0px" }
  );

  revealItems.forEach((item, index) => {
    item.style.transitionDelay = `${Math.min(index * 45, 320)}ms`;
    revealObserver.observe(item);
  });

  const counters = document.querySelectorAll("[data-count]");
  counters.forEach((counter) => {
    const target = Number(counter.getAttribute("data-count"));
    const startRaw = counter.getAttribute("data-animate-from");
    const start = startRaw === null ? target : Number(startRaw);
    if (!Number.isFinite(target) || !Number.isFinite(start) || start === target) {
      counter.textContent = String(target);
      return;
    }

    let current = start;
    const step = Math.max(1, Math.ceil(Math.abs(target - start) / 20));
    const direction = start < target ? 1 : -1;

    const run = () => {
      current += step * direction;
      const done = direction > 0 ? current >= target : current <= target;
      if (done) {
        counter.textContent = String(target);
        return;
      }
      counter.textContent = String(current);
      requestAnimationFrame(run);
    };

    const countObserver = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            counter.textContent = String(start);
            run();
            countObserver.unobserve(counter);
          }
        });
      },
      { threshold: 0.4 }
    );
    countObserver.observe(counter);
  });

  const lightboxTriggers = document.querySelectorAll('[data-lightbox-image]');
  if (lightboxTriggers.length > 0) {
    const lightbox = document.createElement('div');
    lightbox.className = 'lightbox';
    lightbox.setAttribute('role', 'dialog');
    lightbox.setAttribute('aria-modal', 'true');
    lightbox.setAttribute('aria-label', 'Image preview');
    lightbox.innerHTML = `
      <div class="lightbox-panel">
        <figure class="lightbox-image-wrap">
          <img class="lightbox-image" src="" alt="" />
        </figure>
        <div class="lightbox-meta">
          <p class="lightbox-caption"></p>
          <button class="lightbox-close" type="button">Close</button>
        </div>
      </div>
    `;
    document.body.appendChild(lightbox);

    const lightboxImage = lightbox.querySelector('.lightbox-image');
    const lightboxCaption = lightbox.querySelector('.lightbox-caption');
    const closeButton = lightbox.querySelector('.lightbox-close');
    const panel = lightbox.querySelector('.lightbox-panel');
    let lastFocus = null;

    const closeLightbox = () => {
      lightbox.classList.remove('is-open');
      document.body.classList.remove('no-scroll');
      if (lastFocus && typeof lastFocus.focus === 'function') {
        lastFocus.focus();
      }
    };

    const openLightbox = (trigger) => {
      const src = trigger.getAttribute('data-lightbox-image');
      if (!src) return;
      const alt = trigger.getAttribute('data-lightbox-alt') || '';
      const caption = trigger.getAttribute('data-lightbox-caption') || '';
      lightboxImage.setAttribute('src', src);
      lightboxImage.setAttribute('alt', alt);
      lightboxCaption.textContent = caption;
      lastFocus = trigger;
      lightbox.classList.add('is-open');
      document.body.classList.add('no-scroll');
      closeButton.focus();
    };

    lightboxTriggers.forEach((trigger) => {
      trigger.addEventListener('click', () => {
        openLightbox(trigger);
      });
      trigger.addEventListener('keydown', (event) => {
        if (event.key === 'Enter' || event.key === ' ') {
          event.preventDefault();
          openLightbox(trigger);
        }
      });
    });

    closeButton.addEventListener('click', closeLightbox);
    lightbox.addEventListener('click', (event) => {
      if (!panel.contains(event.target)) {
        closeLightbox();
      }
    });
    document.addEventListener('keydown', (event) => {
      if (event.key === 'Escape' && lightbox.classList.contains('is-open')) {
        closeLightbox();
      }
    });
  }
})();
