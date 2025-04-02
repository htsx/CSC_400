document.addEventListener('DOMContentLoaded', function () {
  // Get the dropdown button and dropdown content
  const dropdownBtn = document.querySelector('.dropbtn');
  const dropdownContent = document.querySelector('.dropdown-content');

  // Toggle dropdown visibility on click
  dropdownBtn.addEventListener('click', function() {
    dropdownContent.classList.toggle('show');  // Toggle 'show' class
  });

  // Close dropdown if clicked outside
  window.addEventListener('click', function(event) {
    if (!event.target.matches('.dropbtn') && !event.target.closest('.dropdown')) {
      dropdownContent.classList.remove('show');  // Hide dropdown when clicking outside
    }
  });
});

  // Function to enlarge image when clicked
  function enlargeImage(imgElement) {
    var modal = document.getElementById("imageModal");
    var modalImg = document.getElementById("modalImage");
    modal.style.display = "block";
    modalImg.src = imgElement.src;
  }

  // Function to close the modal when the close button is clicked
  function closeModal() {
    var modal = document.getElementById("imageModal");
    modal.style.display = "none";
  }

  // Close modal when clicking outside of the image
  window.onclick = function(event) {
    var modal = document.getElementById("imageModal");
    if (event.target === modal) {
      modal.style.display = "none";
    }
  }
