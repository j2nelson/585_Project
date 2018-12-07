// In this case, We set width 320, and the height will be computed based on the input stream.
let width = 480;
let height = 0;

// whether streaming video from the camera.
let streaming = false;

let video = document.getElementById("video");
let stream = null;
let vc = null;

function startCamera() {
  if (streaming) return;
  navigator.mediaDevices.getUserMedia({video: true, audio: false})
    .then(function(s) {
    stream = s;
    video.srcObject = s;
    video.play();
  })
    .catch(function(err) {
    console.log("An error occured! " + err);
  });

  video.addEventListener("canplay", function(ev){
    if (!streaming) {
      height = video.videoHeight / (video.videoWidth/width);
      video.setAttribute("width", width);
      video.setAttribute("height", height);
      streaming = true;
      vc = new cv.VideoCapture(video);
    }
    startVideoProcessing();
  }, false);
}

let lastFilter = '';
let src = null;
let dstC1 = null;
let dstC3 = null;
let dstC4 = null;
let canvas = null;
let prevCentroid = null;

function startVideoProcessing() {
  if (!streaming) { console.warn("Please startup your webcam"); return; }
  stopVideoProcessing();
  src = new cv.Mat(height, width, cv.CV_8UC4);
  dstC1 = new cv.Mat(height, width, cv.CV_8UC1);
  dstC3 = new cv.Mat(height, width, cv.CV_8UC3);
  dstC4 = new cv.Mat(height, width, cv.CV_8UC4);
  canvas = new cv.Mat(height, width, cv.CV_8UC1);
  prevCentroid = {x:-1, y:-1};
  requestAnimationFrame(processVideo);
}

function passThrough(src) {
  return src;
}

function gray(src) {
  cv.cvtColor(src, dstC1, cv.COLOR_RGBA2GRAY);
  return dstC1;
}

function hsv(src) {
  cv.cvtColor(src, dstC3, cv.COLOR_RGBA2RGB);
  cv.cvtColor(dstC3, dstC3, cv.COLOR_RGB2HSV);
  return dstC3;
}

function inRange(src) {
  let lowValue = controls.inRangeLow;
  let lowScalar = new cv.Scalar(lowValue, lowValue, lowValue, 255);
  let highValue = controls.inRangeHigh;
  let highScalar = new cv.Scalar(highValue, highValue, highValue, 255);
  let low = new cv.Mat(height, width, src.type(), lowScalar);
  let high = new cv.Mat(height, width, src.type(), highScalar);
  cv.inRange(src, low, high, dstC1);
  low.delete(); high.delete();
  return dstC1;
}

function threshold(src) {
  cv.threshold(src, dstC1, 0.0, 1.0, cv.THRESH_BINARY);
  return dstC1;
}


/**
    Extracts bright regions in image.

    @param src Input image in grayscale
    @return Thresholded binary
*/
function extractBright(src) {
  cv.threshold(src, dstC1, 253, 255, cv.THRESH_BINARY);
  return dstC1;
}

/**
    Gets largest blob in image if area takes up more than 0.1% of image.

    @param src Input image in binary
    @return Binary image with largest blob
*/
function getLargestBlob(src) {
  let contours  = new cv.MatVector();
  let hierarchy = new cv.Mat();
  cv.findContours(src, contours, hierarchy, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE, {x: 0, y: 0});
  let maxsize = 0;
  let maxind = 0;
  for (let i = 0; i<contours.size(); i++) {
    let area = cv.contourArea(contours.get(i), false);
    if( area > maxsize ) {
      maxsize = area;
      maxind = i;
    }
  }
  let percent = maxsize / (width * height);
  let color = new cv.Scalar(255, 255, 255);
  dstC3 = cv.Mat.zeros(height, width, cv.CV_8UC3);
  if(percent > 0.001) cv.drawContours(dstC3, contours, maxind, color, 0, cv.LINE_8, hierarchy);
  contours.delete(); hierarchy.delete();
  return dstC3;
}

/**
    Gets centroid of blob and flips point with respect to y-axis.

    @param src Input image in binary
    @return Centroid of white pixels in binary image
*/
function getMirroredCentroidofBlob(src) {
  let m10, m01, area;
  area = 1;
  m10 = 0;
  m01 = 0;
  for(let y = 0; y < src.rows; y++) {
    for(let x = 0; x < src.cols; x++) {
      if(src.isContinuous()) {
        let gray = src.ucharAt(y, x * src.channels());
        if(gray > 0) {
          area++;
          m10+=x;
          m01+=y;
        }
      }
    }
  }
  let centroid = {x: m10 / area, y: m01 / area};
  let distToCenter = centroid.x - src.cols / 2;
  centroid.x = centroid.x - distToCenter * 2;
  // cv.circle(canvas, centroid, 10, [255, 0, 0, 255], 1, 8, 0);
  return centroid;
}

function getCentroidofBlob(src) {
  let m10, m01, area;
  area = 1;
  m10 = 0;
  m01 = 0;
  for(let y = 0; y < src.rows; y++) {
    for(let x = 0; x < src.cols; x++) {
      if(src.isContinuous()) {
        let gray = src.ucharAt(y, x * src.channels());
        if(gray > 0) {
          area++;
          m10+=x;
          m01+=y;
        }
      }
    }
  }
  let centroid = {x: m10 / area, y: m01 / area};
  return centroid;
}

/**
    Draws line between previous and current tracked point.

    @param curr Current tracked point
    @param prev Tracked point from previous frame
*/
function connectDots(curr, prev) {
  let color = new cv.Scalar(255, 255, 255);
  cv.line(canvas, prev, curr, color, 4, 8, 0);
}

function clearCanvas() {
  canvas = cv.Mat.zeros(height, width, cv.CV_8UC1);
}

function saveCanvas() {
  let centroid = getCentroidofBlob(canvas);
  cv.circle(canvas, centroid, 10, [255, 0, 0, 255], 1, 8, 0);
  let minX = width;
  let maxX = 0;
  let minY = height;
  let maxY  = 0;
  for(let y = 0; y < canvas.rows; y++) {
    for(let x = 0; x < canvas.cols; x++) {
      if(canvas.isContinuous()) {
        let gray = canvas.ucharAt(y, x * canvas.channels());
        if(gray == 255) {
          if(x < minX) minX = x;
          if(y < minY) minY = y;
          if(x > maxX) maxX = x;
          if(y > maxY) maxY = y;
        }
      }
    }
  }

  let boxw = maxX - minX;
  let boxh = maxY - minY;
  let dim = (boxh > boxw)? boxh : (boxw > height)? height : boxw;
  dim = (dim * 1.5 > height)? height : dim * 1.5;
  let left = ((centroid.x + dim/2) > width)? width - dim : ((centroid.x - dim/2) <= 0)? 0 : centroid.x - dim/2;
  let top = ((centroid.y + dim/2) > height)? height - dim : ((centroid.y - dim/2) <= 0)? 0 : centroid.y - dim/2;
  let rect = {x: left, y: top, width: dim, height: dim};
  let character = canvas.roi(rect);

  cv.imshow("characterOutput", character);
}

function getDistanceBetweenPoints(a, b) {
  return Math.sqrt(Math.pow(a.x - b.x, 2) + Math.pow(a.y - b.y, 2));
}

function processVideo() {
  stats.begin();

  // Gets frame from video and extracts bright regions from image
  vc.read(src);
  let process = cv.Mat.zeros(height, width, cv.CV_8UC1);
  process = gray(src);
  process = extractBright(process);

  // Gets largest blob in thresholded image
  process = getLargestBlob(process);

  // Gets center of blob, flip point in y-axis, and draw on canvas
  let centroid = getMirroredCentroidofBlob(process);
  if(getDistanceBetweenPoints(prevCentroid, centroid) < 100 && prevCentroid.x > 0 && prevCentroid.y > 0 && centroid.x > 0 && centroid.y > 0)
    connectDots(centroid, prevCentroid);

  cv.imshow("processOutput", process);
  cv.imshow("canvasOutput", canvas);
  stats.end();
  prevCentroid = centroid;

  requestAnimationFrame(processVideo);
}

function stopVideoProcessing() {
  if (src != null && !src.isDeleted()) src.delete();
  if (dstC1 != null && !dstC1.isDeleted()) dstC1.delete();
  if (dstC3 != null && !dstC3.isDeleted()) dstC3.delete();
  if (dstC4 != null && !dstC4.isDeleted()) dstC4.delete();
}

function stopCamera() {
  if (!streaming) return;
  stopVideoProcessing();
  document.getElementById("canvasOutput").getContext("2d").clearRect(0, 0, width, height);
  document.getElementById("processOutput").getContext("2d").clearRect(0, 0, width, height);
  document.getElementById("characterOutput").getContext("2d").clearRect(0, 0, width, height);
  video.pause();
  video.srcObject=null;
  stream.getVideoTracks()[0].stop();
  streaming = false;
}

var stats = null;

var controls;

function initUI() {
  stats = new Stats();
  stats.showPanel(0);
  document.getElementById('container').appendChild(stats.domElement);
}

function opencvIsReady() {
  console.log('OpenCV.js is ready');
  initUI();
  startCamera();
  window.addEventListener('keypress', clearCanvas);
  window.addEventListener('click', saveCanvas);
}
