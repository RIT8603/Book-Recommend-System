document.addEventListener("DOMContentLoaded", () => {
    document.querySelectorAll(".loading-button").forEach((button) => {
        button.closest("form").addEventListener("submit", (event) => {
            if (button.closest(".retrain-form")) {
                const confirmed = window.confirm("Retraining replaces books.pkl, popular.pkl, pt.pkl, and similarity_scores.pkl. Continue?");
                if (!confirmed) {
                    event.preventDefault();
                    return;
                }
            }

            if (button.closest(".import-form")) {
                const confirmed = window.confirm("This may send many rows to Supabase and can take a while. Continue?");
                if (!confirmed) {
                    event.preventDefault();
                    return;
                }
            }

            const text = button.querySelector(".button-text");
            const spinner = button.querySelector(".spinner-border");

            if (text && spinner) {
                text.textContent = "Loading...";
                spinner.classList.remove("d-none");
            }

            button.disabled = true;
        });
    });

    document.querySelectorAll(".delete-form").forEach((form) => {
        form.addEventListener("submit", (event) => {
            const confirmed = window.confirm("Delete this book from Supabase?");
            if (!confirmed) {
                event.preventDefault();
            }
        });
    });

    const autocompleteInput = document.querySelector(".autocomplete-input");
    const suggestionList = document.getElementById("bookSuggestions");

    if (autocompleteInput && suggestionList) {
        let autocompleteTimer;

        autocompleteInput.addEventListener("input", () => {
            window.clearTimeout(autocompleteTimer);
            const query = autocompleteInput.value.trim();

            if (query.length < 2) {
                suggestionList.innerHTML = "";
                return;
            }

            autocompleteTimer = window.setTimeout(async () => {
                try {
                    const response = await fetch(`/api/suggestions?q=${encodeURIComponent(query)}`);
                    const suggestions = await response.json();
                    suggestionList.innerHTML = suggestions
                        .map((item) => `<option value="${item.title}">${item.score}% match</option>`)
                        .join("");
                } catch (_error) {
                    suggestionList.innerHTML = "";
                }
            }, 180);
        });
    }

    document.querySelectorAll(".book-cover").forEach((image) => {
        image.addEventListener("error", () => {
            image.src = "https://placehold.co/320x480/111827/f8fafc?text=Book+Cover";
        });
    });
});
