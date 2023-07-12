document.addEventListener('DOMContentLoaded', function() {
  const videos = document.querySelectorAll('.background-video');
  let currentVideo = 0;

  setInterval(function() {
    videos[currentVideo].style.opacity = '0';

    currentVideo++;
    if (currentVideo >= videos.length) {
      currentVideo = 0;
    }

    videos[currentVideo].style.opacity = '1';
  }, 5000);

});
function login() {
  // Perform login validation
  var username = document.getElementById("username").value;
  var password = document.getElementById("password").value;

  // Example validation: check if username and password are correct
  if (username === "exampleuser" && password === "examplepassword") {
    // Redirect to the recognize page
    window.location.href = "recognize.html";
    return false; // Prevent form submission
  } else {
    alert("Invalid username or password.");
    return false; // Prevent form submission
  }
}
function recognizeFace() {
  // Code for face recognition
}

function recognizeVoice() {
  // Code for voice recognition
}

var users = []; // Array to store user data


function signup() {
  var username = document.getElementById("username").value;
  var password = document.getElementById("password").value;

  // Check if the username already exists
  if (checkUserExists(username)) {
    alert("Username already exists. Please choose a different username.");
    return false; // Prevent form submission
  }

  // Add the new user to the array
  users.push({ username: username, password: password });

  // Redirect to the record page
  window.location.href = "record.html";
  return false; // Prevent form submission
}

function checkUserExists(username) {
  // Check if the username already exists in the users array
  return users.some(function(user) {
    return user.username === username;
  });
}

function captureImages() {
  // Code for capturing images
}

function recordVoice() {
  // Code for recording voice
}

if (window.location.href.endsWith('recognized.html')) {
  const recognizedContainer = document.querySelector('.recognized-container');
  recognizedContainer.style.display = 'flex';
  recognizedContainer.style.justifyContent = 'center';
  recognizedContainer.style.alignItems = 'center';
}
