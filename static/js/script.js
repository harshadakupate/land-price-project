/* =========================================
   LAND AI - MASTER JS
   ========================================= */

/* Navigation */
function go(path) {
    window.location.href = path;
}

/* Budget Slider */
function updateValue(val) {
    let display = document.getElementById("budgetValue");
    if (display) {
        let num = parseInt(val).toLocaleString("en-IN");
        display.innerText = num;
    }
}

/* Card hover glow */
document.addEventListener("DOMContentLoaded", function () {
    document.querySelectorAll(".card").forEach(card => {
        card.addEventListener("mouseenter", () => {
            card.style.boxShadow = "0 12px 40px rgba(0,198,255,0.25)";
        });
        card.addEventListener("mouseleave", () => {
            card.style.boxShadow = "0 8px 32px rgba(0,0,0,0.4)";
        });
    });
});
