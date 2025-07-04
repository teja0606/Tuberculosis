// static/js/main.js

document.addEventListener("DOMContentLoaded", () => {
  // --- THEME TOGGLE LOGIC (GLOBAL) ---
  const themeToggle = document.getElementById("theme-toggle");
  if (themeToggle) {
    const currentTheme = localStorage.getItem("theme") || "light";
    document.documentElement.setAttribute("data-theme", currentTheme);

    themeToggle.addEventListener("click", () => {
      let newTheme =
        document.documentElement.getAttribute("data-theme") === "dark"
          ? "light"
          : "dark";
      document.documentElement.setAttribute("data-theme", newTheme);
      localStorage.setItem("theme", newTheme);
    });
  }

  // --- CHATBOT PAGE LOGIC ---
  // FIX: Changed the selector to a valid ID on the chatbot page (#chat-box)
  const chatBox = document.getElementById("chat-box");
  if (chatBox) {
    const suggestionCardsContainer =
      document.getElementById("suggestion-cards");
    const userInput = document.getElementById("user-input");
    const sendButton = document.getElementById("send-button");

    const suggestions = [
      "What is Tuberculosis?",
      "How is TB transmitted?",
      "What are the common symptoms?",
      "How can TB be prevented?",
    ];

    function initChat() {
      addMessage(
        "Hello! I'm Aura, your AI assistant. I can provide general information about Tuberculosis. How can I help?",
        "bot"
      );
      renderSuggestionCards();
    }

    function renderSuggestionCards() {
      suggestionCardsContainer.innerHTML = "";
      suggestions.forEach((text) => {
        const card = document.createElement("div");
        card.classList.add("suggestion-card");
        card.textContent = text;
        card.addEventListener("click", () => handleSuggestionClick(text));
        suggestionCardsContainer.appendChild(card);
      });
    }

    function handleSuggestionClick(text) {
      userInput.value = text;
      sendMessage();
    }

    function addMessage(text, sender) {
      const messageDiv = document.createElement("div");
      messageDiv.classList.add("chat-message", sender);

      const avatar = document.createElement("div");
      avatar.classList.add("avatar");
      avatar.innerHTML =
        sender === "bot"
          ? '<i class="ph-bold ph-robot"></i>'
          : '<i class="ph-bold ph-user"></i>';

      const messageContent = document.createElement("div");
      messageContent.classList.add("message-content");

      if (sender === "bot" && text === "...") {
        messageDiv.classList.add("typing");
        messageContent.innerHTML = "<p>Aura is typing...</p>";
      } else {
        // Use marked.js to render markdown from bot
        messageContent.innerHTML =
          sender === "bot" ? marked.parse(text) : `<p>${text}</p>`;
      }

      messageDiv.appendChild(avatar);
      messageDiv.appendChild(messageContent);
      chatBox.appendChild(messageDiv);
      chatBox.scrollTop = chatBox.scrollHeight;
    }

    function sendMessage() {
      const message = userInput.value.trim();
      if (!message) return;

      addMessage(message, "user");
      userInput.value = "";
      suggestionCardsContainer.style.display = "none";
      addMessage("...", "bot");

      fetch("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message }),
      })
        .then((response) => response.json())
        .then((data) => {
          const typingIndicator = chatBox.querySelector(".typing");
          if (typingIndicator) typingIndicator.remove();
          const reply = data.error ? `Error: ${data.error}` : data.reply;
          addMessage(reply, "bot");
        })
        .catch((error) => {
          console.error("Error:", error);
          const typingIndicator = chatBox.querySelector(".typing");
          if (typingIndicator) typingIndicator.remove();
          addMessage("Sorry, I encountered an error.", "bot");
        });
    }

    sendButton.addEventListener("click", sendMessage);
    userInput.addEventListener("keypress", (e) => {
      if (e.key === "Enter") {
        sendMessage();
      }
    });

    initChat();
  }

  // --- DETECTOR PAGE LOGIC ---
  const detectorWrapper = document.querySelector(".detector-wrapper");
  if (detectorWrapper) {
    const uploadContainer = document.getElementById("upload-container");
    const resultsContainer = document.getElementById("results-container");
    const uploadForm = document.getElementById("upload-form");
    const uploadArea = document.getElementById("upload-area");
    const fileInput = document.getElementById("file-input");
    const imagePreview = document.getElementById("image-preview");
    const predictionText = document.getElementById("prediction-text");
    const confidenceScore = document.getElementById("confidence-score");
    const analyzeAnotherBtn = document.getElementById("analyze-another-btn");

    uploadArea.addEventListener("click", () => fileInput.click());
    fileInput.addEventListener("change", () => {
      if (fileInput.files.length > 0) {
        handleFile(fileInput.files[0]);
      }
    });

    ["dragenter", "dragover", "dragleave", "drop"].forEach((eventName) => {
      uploadArea.addEventListener(eventName, preventDefaults, false);
    });
    function preventDefaults(e) {
      e.preventDefault();
      e.stopPropagation();
    }
    ["dragenter", "dragover"].forEach((eventName) => {
      uploadArea.addEventListener(
        eventName,
        () => uploadArea.classList.add("dragover"),
        false
      );
    });
    ["dragleave", "drop"].forEach((eventName) => {
      uploadArea.addEventListener(
        eventName,
        () => uploadArea.classList.remove("dragover"),
        false
      );
    });
    uploadArea.addEventListener("drop", (e) => {
      if (e.dataTransfer.files.length > 0) {
        handleFile(e.dataTransfer.files[0]);
      }
    });

    function handleFile(file) {
      uploadContainer.style.display = "none";
      resultsContainer.style.display = "flex";

      const reader = new FileReader();
      reader.onload = (e) => {
        imagePreview.src = e.target.result;
      };
      reader.readAsDataURL(file);

      const formData = new FormData();
      formData.append("file", file);

      // Show loading state
      predictionText.textContent = "Analyzing...";
      predictionText.className = "";
      confidenceScore.textContent = "---";

      fetch("/predict", { method: "POST", body: formData })
        .then((response) => response.json())
        .then((data) => {
          if (data.error) {
            alert(`Error: ${data.error}`);
            resetDetectorUI();
            return;
          }
          displayDetectorResults(data);
        })
        .catch((error) => {
          console.error("Error:", error);
          alert("An unexpected error occurred.");
          resetDetectorUI();
        });
    }

    function displayDetectorResults(data) {
      const prediction = data.prediction;
      const confidence = parseFloat(data.confidence);

      predictionText.textContent = prediction;
      confidenceScore.textContent = `${(confidence * 100).toFixed(1)}%`;

      const predictionClass = prediction.toLowerCase();
      predictionText.className = "";
      predictionText.classList.add(predictionClass);
    }

    function resetDetectorUI() {
      uploadContainer.style.display = "block";
      resultsContainer.style.display = "none";
      uploadForm.reset();
    }

    analyzeAnotherBtn.addEventListener("click", resetDetectorUI);
  }
});
