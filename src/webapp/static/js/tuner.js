(async function () {
  const el = (id) => document.getElementById(id);

  const video = el('video'), c0 = el('c0'),
        orig = el('orig'), enhColor = el('enhColor'),
        enhGray = el('enhGray'), binary = el('binary');

  let ctx0 = c0.getContext('2d', { willReadFrequently: true });
  let ctxOrig = orig.getContext('2d', { willReadFrequently: true });
  let ctxEnhC = enhColor.getContext('2d', { willReadFrequently: true });
  let ctxEnhG = enhGray.getContext('2d', { willReadFrequently: true });
  let ctxBin  = binary.getContext('2d', { willReadFrequently: true });

  let flip = false;

  const sliders = {
    alpha: el('alpha'),
    beta: el('beta'),
    gamma: el('gamma'),
    sat: el('sat'),
    sharp: el('sharp'),
    clahe: el('clahe'),
    thr: el('thr'),
    useOtsu: el('useOtsu'),
  };
  const labels = {
    alpha: el('alphaVal'),
    beta: el('betaVal'),
    gamma: el('gammaVal'),
    sat: el('satVal'),
    sharp: el('sharpVal'),
    clahe: el('claheVal'),
    thr: el('thrVal'),
  };

  function updateLabels() {
    labels.alpha.textContent = (Number(sliders.alpha.value)/100).toFixed(2);
    labels.beta.textContent = sliders.beta.value;
    labels.gamma.textContent = (Number(sliders.gamma.value)/100).toFixed(2);
    labels.sat.textContent = sliders.sat.value + '%';
    labels.sharp.textContent = sliders.sharp.value + '%';
    labels.clahe.textContent = (Number(sliders.clahe.value)/100).toFixed(2);
    labels.thr.textContent = sliders.thr.value;
  }
  Object.values(sliders).forEach(s => s.addEventListener('input', updateLabels));
  updateLabels();

  // Resolution
  async function setResolution() {
    const [w, h] = el('resolution').value.split('x').map(Number);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: w }, height: { ideal: h },
          facingMode: 'environment'
        }, audio: false
      });
      if (video.srcObject) {
        video.srcObject.getTracks().forEach(t => t.stop());
      }
      video.srcObject = stream;
    } catch (e) {
      alert('Camera error: ' + e);
    }
  }
  el('resolution').addEventListener('change', setResolution);
  el('flip').addEventListener('click', () => flip = !flip);
  el('snap').addEventListener('click', () => {
    const a = document.createElement('a');
    const grid = document.createElement('canvas');
    const gctx = grid.getContext('2d');
    const w = enhColor.width, h = enhColor.height;
    grid.width = w*2; grid.height = h*2;
    gctx.drawImage(orig, 0, 0);
    gctx.drawImage(enhColor, w, 0);
    gctx.drawImage(enhGray, 0, h);
    gctx.drawImage(binary, w, h);
    a.href = grid.toDataURL('image/png');
    a.download = `tuner_${Date.now()}.png`;
    a.click();
  });

  // Start camera initially
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    alert("Your browser doesn't support getUserMedia. Use Chrome/Safari/Firefox over HTTPS.");
    return;
  }
  await setResolution();

  // Image ops -------------------------------------------------------

  function buildGammaLUT(gamma) {
    gamma = Math.max(0.1, Math.min(5.0, gamma));
    const lut = new Uint8ClampedArray(256);
    for (let i=0;i<256;i++) lut[i] = Math.max(0, Math.min(255, Math.pow(i/255, gamma)*255));
    return lut;
  }

  function rgb2hsv(r,g,b){
    const max = Math.max(r,g,b), min = Math.min(r,g,b);
    const v = max, d = max - min;
    const s = max === 0 ? 0 : d / max;
    let h = 0;
    if (max !== min) {
      switch (max) {
        case r: h = (g - b) / d + (g < b ? 6 : 0); break;
        case g: h = (b - r) / d + 2; break;
        case b: h = (r - g) / d + 4; break;
      }
      h /= 6;
    }
    return [h, s, v/255];
  }

  function adjustSaturation(imgData, satScale){
    if (Math.abs(satScale - 1.0) < 1e-3) return;
    const d = imgData.data;
    for (let i=0;i<d.length;i+=4){
      let r=d[i], g=d[i+1], b=d[i+2];
      const [h,s,v] = rgb2hsv(r,g,b);
      const newS = Math.max(0, Math.min(1, s*satScale));
      // hsv->rgb (simple)
      const C = v*newS;
      const X = C*(1-Math.abs(((h*6)%2)-1));
      const m = v-C;
      let rp=0,gp=0,bp=0;
      const hi = Math.floor(h*6);
      if (hi===0){ rp=C; gp=X; bp=0; }
      else if (hi===1){ rp=X; gp=C; bp=0; }
      else if (hi===2){ rp=0; gp=C; bp=X; }
      else if (hi===3){ rp=0; gp=X; bp=C; }
      else if (hi===4){ rp=X; gp=0; bp=C; }
      else { rp=C; gp=0; bp=X; }
      d[i]  = Math.round((rp+m)*255);
      d[i+1]= Math.round((gp+m)*255);
      d[i+2]= Math.round((bp+m)*255);
    }
  }

  function unsharp(imgData, amount){
    if (amount <= 1e-6) return;
    const {width, height, data} = imgData;
    // simple Gaussian blur via separable box (approx)
    function boxBlur(src, w, h, r){
      const tmp = new Uint8ClampedArray(src.length);
      const out = new Uint8ClampedArray(src.length);
      const ch = 4;
      // horizontal
      for(let y=0;y<h;y++){
        for(let x=0;x<w;x++){
          for(let c=0;c<3;c++){
            let acc=0, cnt=0;
            for(let k=-r;k<=r;k++){
              const xx = Math.min(w-1, Math.max(0, x+k));
              acc += src[(y*w+xx)*ch+c];
              cnt++;
            }
            tmp[(y*w+x)*ch+c] = acc/cnt;
          }
          tmp[(y*w+x)*ch+3] = src[(y*w+x)*ch+3];
        }
      }
      // vertical
      for(let y=0;y<h;y++){
        for(let x=0;x<w;x++){
          for(let c=0;c<3;c++){
            let acc=0, cnt=0;
            for(let k=-r;k<=r;k++){
              const yy = Math.min(h-1, Math.max(0, y+k));
              acc += tmp[(yy*w+x)*ch+c];
              cnt++;
            }
            out[(y*w+x)*ch+c] = acc/cnt;
          }
          out[(y*w+x)*ch+3] = src[(y*w+x)*ch+3];
        }
      }
      return out;
    }
    const blurred = boxBlur(data, imgData.width, imgData.height, 2);
    for(let i=0;i<data.length;i+=4){
      data[i]   = Math.max(0, Math.min(255, data[i]*(1+amount) - blurred[i]*amount));
      data[i+1] = Math.max(0, Math.min(255, data[i+1]*(1+amount) - blurred[i+1]*amount));
      data[i+2] = Math.max(0, Math.min(255, data[i+2]*(1+amount) - blurred[i+2]*amount));
    }
  }

  function toGray(imgData){
    const d = imgData.data;
    for (let i=0;i<d.length;i+=4){
      const y = (0.299*d[i] + 0.587*d[i+1] + 0.114*d[i+2])|0;
      d[i]=d[i+1]=d[i+2]=y;
    }
  }

  function claheLike(grayData, clip){ // super light "local contrast" approx
    if (clip <= 0) return;
    // simple local normalization over a 8x8 tile neighborhood (fast-ish)
    const w = grayData.width, h = grayData.height, d = grayData.data;
    const tile = 8, half = 4;
    // integral image
    const sum = new Float32Array((w+1)*(h+1));
    for(let y=1;y<=h;y++){
      let row=0;
      for(let x=1;x<=w;x++){
        row += d[((y-1)*w+(x-1))*4];
        sum[y*(w+1)+x] = sum[(y-1)*(w+1)+x] + row;
      }
    }
    function rectSum(x0,y0,x1,y1){
      x0=Math.max(0,x0); y0=Math.max(0,y0);
      x1=Math.min(w,x1); y1=Math.min(h,y1);
      return sum[y1*(w+1)+x1] - sum[y0*(w+1)+x1] - sum[y1*(w+1)+x0] + sum[y0*(w+1)+x0];
    }
    const area = tile*tile;
    for(let y=0;y<h;y++){
      for(let x=0;x<w;x++){
        const x0=x-half, y0=y-half, x1=x+half, y1=y+half;
        const m = rectSum(x0,y0,x1,y1)/area;
        const idx=(y*w+x)*4;
        const val=d[idx]-m; // zero-mean
        d[idx]=d[idx+1]=d[idx+2]=Math.max(0, Math.min(255, val*1.2 + 128));
      }
    }
  }

  function threshold(grayData, thr, useOtsu){
    const w=grayData.width,h=grayData.height,d=grayData.data;
    if (useOtsu){
      // simple Otsu: histogram + between-class variance
      const hist = new Uint32Array(256);
      for(let i=0;i<d.length;i+=4) hist[d[i]]++;
      let sum=0,sumB=0,wB=0,wF=0,mB=0,mF=0,max=0,th=0,total=w*h;
      for(let t=0;t<256;t++) sum += t*hist[t];
      for(let t=0;t<256;t++){
        wB += hist[t]; if (wB===0) continue;
        wF = total - wB; if (wF===0) break;
        sumB += t*hist[t];
        mB = sumB / wB; mF = (sum - sumB) / wF;
        const between = wB*wF*(mB-mF)*(mB-mF);
        if (between > max){ max = between; th = t; }
      }
      thr = th;
    }
    for(let i=0;i<d.length;i+=4){
      const v = d[i] > thr ? 255 : 0;
      d[i]=d[i+1]=d[i+2]=v;
    }
  }

  function render() {
    const stream = video.srcObject;
    if (!stream) { requestAnimationFrame(render); return; }
    const vw = video.videoWidth || 640, vh = video.videoHeight || 480;
    [c0, orig, enhColor, enhGray, binary].forEach(c => { c.width = vw; c.height = vh; });

    // Draw original
    ctx0.save();
    ctx0.setTransform(flip?-1:1, 0, 0, 1, flip?vw:0, 0);
    ctx0.drawImage(video, 0, 0, vw, vh);
    ctx0.restore();

    const frame = ctx0.getImageData(0, 0, vw, vh);

    // 1) Brightness/Contrast
    const alpha = Number(sliders.alpha.value)/100.0;
    const beta  = Number(sliders.beta.value)|0;
    const d1 = new ImageData(new Uint8ClampedArray(frame.data), vw, vh);
    for(let i=0;i<d1.data.length;i+=4){
      d1.data[i]   = Math.min(255, Math.max(0, (d1.data[i]  * alpha + beta)));
      d1.data[i+1] = Math.min(255, Math.max(0, (d1.data[i+1]* alpha + beta)));
      d1.data[i+2] = Math.min(255, Math.max(0, (d1.data[i+2]* alpha + beta)));
    }

    // 2) Gamma
    const gamma = Math.max(0.1, Number(sliders.gamma.value)/100.0);
    const lut = (function(){ const l = new Uint8ClampedArray(256); for(let i=0;i<256;i++) l[i] = Math.pow(i/255, gamma)*255; return l; })();
    for(let i=0;i<d1.data.length;i+=4){
      d1.data[i]   = lut[d1.data[i]];
      d1.data[i+1] = lut[d1.data[i+1]];
      d1.data[i+2] = lut[d1.data[i+2]];
    }

    // 3) Saturation
    const satScale = Math.max(0, Number(sliders.sat.value)/100.0);
    adjustSaturation(d1, satScale);

    // 4) Unsharp
    const sharpAmt = Math.max(0, Math.min(2.0, Number(sliders.sharp.value)/100.0));
    unsharp(d1, sharpAmt);

    // Enhanced color
    ctxEnhC.putImageData(d1, 0, 0);

    // 5) Gray (+light CLAHE-ish)
    const dGray = new ImageData(new Uint8ClampedArray(d1.data), vw, vh);
    toGray(dGray);
    const clahe = Number(sliders.clahe.value)/100.0;
    if (clahe > 0) claheLike(dGray, clahe);
    ctxEnhG.putImageData(dGray, 0, 0);

    // 6) Threshold
    const useOtsu = sliders.useOtsu.checked;
    const thr = Number(sliders.thr.value)|0;
    const dBin = new ImageData(new Uint8ClampedArray(dGray.data), vw, vh);
    threshold(dBin, thr, useOtsu);
    ctxBin.putImageData(dBin, 0, 0);

    // Original (top-left)
    ctxOrig.putImageData(frame, 0, 0);

    requestAnimationFrame(render);
  }

  render();
})();
