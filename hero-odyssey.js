/* ============================================================
   Hero Odyssey — vanilla JS port of hero-odyssey.tsx
   - Lightning: WebGL fragment shader (ported verbatim from React)
   - ElasticHueSlider: native range wired to a custom animated thumb
   - Mobile menu toggle + entrance animations (CSS-driven)
   ============================================================ */
(function () {
  "use strict";

  /* ---------------------------------------------------------
     Lightning — WebGL animated shader
     Mirrors the <Lightning /> React component 1:1.
     --------------------------------------------------------- */
  function initLightning(canvas, opts) {
    var hue = opts.hue,
      xOffset = opts.xOffset || 0,
      speed = opts.speed || 1,
      intensity = opts.intensity || 1,
      size = opts.size || 1;

    function resizeCanvas() {
      canvas.width = canvas.clientWidth;
      canvas.height = canvas.clientHeight;
    }
    resizeCanvas();
    window.addEventListener("resize", resizeCanvas);

    var gl = canvas.getContext("webgl");
    if (!gl) {
      console.error("WebGL not supported");
      return { setHue: function () {} };
    }

    var vertexShaderSource =
      "attribute vec2 aPosition;" +
      "void main() { gl_Position = vec4(aPosition, 0.0, 1.0); }";

    var fragmentShaderSource = [
      "precision mediump float;",
      "uniform vec2 iResolution;",
      "uniform float iTime;",
      "uniform float uHue;",
      "uniform float uXOffset;",
      "uniform float uSpeed;",
      "uniform float uIntensity;",
      "uniform float uSize;",
      "#define OCTAVE_COUNT 10",
      "vec3 hsv2rgb(vec3 c) {",
      "  vec3 rgb = clamp(abs(mod(c.x * 6.0 + vec3(0.0,4.0,2.0), 6.0) - 3.0) - 1.0, 0.0, 1.0);",
      "  return c.z * mix(vec3(1.0), rgb, c.y);",
      "}",
      "float hash11(float p) {",
      "  p = fract(p * .1031);",
      "  p *= p + 33.33;",
      "  p *= p + p;",
      "  return fract(p);",
      "}",
      "float hash12(vec2 p) {",
      "  vec3 p3 = fract(vec3(p.xyx) * .1031);",
      "  p3 += dot(p3, p3.yzx + 33.33);",
      "  return fract((p3.x + p3.y) * p3.z);",
      "}",
      "mat2 rotate2d(float theta) {",
      "  float c = cos(theta);",
      "  float s = sin(theta);",
      "  return mat2(c, -s, s, c);",
      "}",
      "float noise(vec2 p) {",
      "  vec2 ip = floor(p);",
      "  vec2 fp = fract(p);",
      "  float a = hash12(ip);",
      "  float b = hash12(ip + vec2(1.0, 0.0));",
      "  float c = hash12(ip + vec2(0.0, 1.0));",
      "  float d = hash12(ip + vec2(1.0, 1.0));",
      "  vec2 t = smoothstep(0.0, 1.0, fp);",
      "  return mix(mix(a, b, t.x), mix(c, d, t.x), t.y);",
      "}",
      "float fbm(vec2 p) {",
      "  float value = 0.0;",
      "  float amplitude = 0.5;",
      "  for (int i = 0; i < OCTAVE_COUNT; ++i) {",
      "    value += amplitude * noise(p);",
      "    p *= rotate2d(0.45);",
      "    p *= 2.0;",
      "    amplitude *= 0.5;",
      "  }",
      "  return value;",
      "}",
      "void mainImage( out vec4 fragColor, in vec2 fragCoord ) {",
      "  vec2 uv = fragCoord / iResolution.xy;",
      "  uv = 2.0 * uv - 1.0;",
      "  uv.x *= iResolution.x / iResolution.y;",
      "  uv.x += uXOffset;",
      "  uv += 2.0 * fbm(uv * uSize + 0.8 * iTime * uSpeed) - 1.0;",
      "  float dist = abs(uv.x);",
      "  vec3 baseColor = hsv2rgb(vec3(uHue / 360.0, 0.7, 0.8));",
      "  vec3 col = baseColor * pow(mix(0.0, 0.07, hash11(iTime * uSpeed)) / dist, 1.0) * uIntensity;",
      "  col = pow(col, vec3(1.0));",
      "  fragColor = vec4(col, 1.0);",
      "}",
      "void main() { mainImage(gl_FragColor, gl_FragCoord.xy); }"
    ].join("\n");

    function compileShader(source, type) {
      var shader = gl.createShader(type);
      if (!shader) return null;
      gl.shaderSource(shader, source);
      gl.compileShader(shader);
      if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        console.error("Shader compile error:", gl.getShaderInfoLog(shader));
        gl.deleteShader(shader);
        return null;
      }
      return shader;
    }

    var vertexShader = compileShader(vertexShaderSource, gl.VERTEX_SHADER);
    var fragmentShader = compileShader(fragmentShaderSource, gl.FRAGMENT_SHADER);
    if (!vertexShader || !fragmentShader) return { setHue: function () {} };

    var program = gl.createProgram();
    if (!program) return { setHue: function () {} };
    gl.attachShader(program, vertexShader);
    gl.attachShader(program, fragmentShader);
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      console.error("Program linking error:", gl.getProgramInfoLog(program));
      return { setHue: function () {} };
    }
    gl.useProgram(program);

    var vertices = new Float32Array([-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1]);
    var vertexBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, vertexBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.STATIC_DRAW);

    var aPosition = gl.getAttribLocation(program, "aPosition");
    gl.enableVertexAttribArray(aPosition);
    gl.vertexAttribPointer(aPosition, 2, gl.FLOAT, false, 0, 0);

    var iResolutionLocation = gl.getUniformLocation(program, "iResolution");
    var iTimeLocation = gl.getUniformLocation(program, "iTime");
    var uHueLocation = gl.getUniformLocation(program, "uHue");
    var uXOffsetLocation = gl.getUniformLocation(program, "uXOffset");
    var uSpeedLocation = gl.getUniformLocation(program, "uSpeed");
    var uIntensityLocation = gl.getUniformLocation(program, "uIntensity");
    var uSizeLocation = gl.getUniformLocation(program, "uSize");

    var startTime = performance.now();
    var rafId;
    function render() {
      resizeCanvas();
      gl.viewport(0, 0, canvas.width, canvas.height);
      gl.uniform2f(iResolutionLocation, canvas.width, canvas.height);
      var currentTime = performance.now();
      gl.uniform1f(iTimeLocation, (currentTime - startTime) / 1000.0);
      gl.uniform1f(uHueLocation, hue);
      gl.uniform1f(uXOffsetLocation, xOffset);
      gl.uniform1f(uSpeedLocation, speed);
      gl.uniform1f(uIntensityLocation, intensity);
      gl.uniform1f(uSizeLocation, size);
      gl.drawArrays(gl.TRIANGLES, 0, 6);
      rafId = requestAnimationFrame(render);
    }
    rafId = requestAnimationFrame(render);

    return {
      setHue: function (h) { hue = h; },
      destroy: function () {
        cancelAnimationFrame(rafId);
        window.removeEventListener("resize", resizeCanvas);
      }
    };
  }

  /* ---------------------------------------------------------
     ElasticHueSlider — native range + animated custom thumb/fill
     Replaces framer-motion spring with a CSS spring transition.
     --------------------------------------------------------- */
  function initHueSlider(root, onChange) {
    var input = root.querySelector(".hue-slider__input");
    var fill = root.querySelector(".hue-slider__fill");
    var thumb = root.querySelector(".hue-slider__thumb");
    var valueEl = root.querySelector(".hue-slider__value");

    var min = Number(input.min) || 0;
    var max = Number(input.max) || 360;

    function update() {
      var value = Number(input.value);
      var progress = (value - min) / (max - min);
      var pct = progress * 100;
      fill.style.width = pct + "%";
      thumb.style.left = pct + "%";
      // retrigger the pop animation on the value label
      valueEl.textContent = value + "°";
      valueEl.style.animation = "none";
      /* force reflow so the animation restarts */
      void valueEl.offsetWidth;
      valueEl.style.animation = "";
      if (onChange) onChange(value);
    }

    input.addEventListener("input", update);

    var startDrag = function () { thumb.classList.add("is-dragging"); };
    var endDrag = function () { thumb.classList.remove("is-dragging"); };
    input.addEventListener("mousedown", startDrag);
    input.addEventListener("touchstart", startDrag, { passive: true });
    window.addEventListener("mouseup", endDrag);
    window.addEventListener("touchend", endDrag);

    update(); // initialise positions
  }

  /* ---------------------------------------------------------
     Mobile menu toggle
     --------------------------------------------------------- */
  function initMobileMenu(root) {
    var toggle = root.querySelector(".hero__menu-toggle");
    var menu = root.querySelector(".hero__mobile-menu");
    var closeBtn = root.querySelector(".hero__mobile-close");
    var openIcon = toggle.querySelector('[data-icon="open"]');
    var closeIcon = toggle.querySelector('[data-icon="close"]');

    function setOpen(open) {
      menu.classList.toggle("is-open", open);
      if (openIcon && closeIcon) {
        openIcon.style.display = open ? "none" : "";
        closeIcon.style.display = open ? "" : "none";
      }
    }
    toggle.addEventListener("click", function () {
      setOpen(!menu.classList.contains("is-open"));
    });
    if (closeBtn) closeBtn.addEventListener("click", function () { setOpen(false); });
  }

  /* --------------------------------------------------------- */
  function boot() {
    var root = document.querySelector(".hero");
    if (!root) return;

    var canvas = root.querySelector(".hero__lightning-canvas");
    var lightning = initLightning(canvas, {
      hue: 220,       // default lightningHue
      xOffset: 0,
      speed: 1.6,
      intensity: 0.6,
      size: 2
    });

    var slider = root.querySelector(".hue-slider");
    if (slider) {
      initHueSlider(slider, function (value) { lightning.setHue(value); });
    }

    initMobileMenu(root);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
