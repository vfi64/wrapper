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
})();
