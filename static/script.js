document.getElementById("plantImage").addEventListener("change", async function() {
    const file = this.files[0];

    if (!file) {
        alert("Please select a file first.");
        return;
    }

    let formData = new FormData();
    formData.append("file", file);

    try {
        const response = await fetch("http://127.0.0.1:5000/predict", {
            method: "POST",
            body: formData
        });

        const data = await response.json();  // Parse response as JSON

        if (data.error) {
            alert("Error: " + data.error);
            return;
        }

        // Display disease and suggestion on the webpage
        document.getElementById("disease").textContent = "Disease: " + (data.disease || "Unknown");
        document.getElementById("suggestion").textContent = "Suggestion: " + (data.suggestion || "No suggestions available.");

    } catch (error) {
        console.error("Error:", error);
        alert("An error occurred while processing the image.");
    }
});
