

pop_up = """
function() {
    setTimeout(function() {
        // Create a more stylish popup
        const popup = document.createElement('div');
        popup.style.position = 'fixed';
        popup.style.top = '20px';
        popup.style.left = '50%';
        popup.style.transform = 'translateX(-50%)';
        popup.style.backgroundColor = '#8C1675';
        popup.style.color = 'white';
        popup.style.padding = '15px 25px';
        popup.style.borderRadius = '10px';
        popup.style.boxShadow = '0 4px 12px rgba(0,0,0,0.2)';
        popup.style.zIndex = '9999';
        popup.style.fontFamily = 'Arial, sans-serif';
        popup.style.fontSize = '16px';
        popup.style.fontWeight = 'bold';
        popup.style.animation = 'fadeIn 0.5s, fadeOut 0.5s 2.5s forwards';
        
        // Add animation keyframes
        const style = document.createElement('style');
        style.textContent = `
            @keyframes fadeIn {
                from { opacity: 0; top: 0; }
                to { opacity: 1; top: 20px; }
            }
            @keyframes fadeOut {
                from { opacity: 1; top: 20px; }
                to { opacity: 0; top: 0; }
            }
        `;
        document.head.appendChild(style);
        
        popup.textContent = '🍏 WELCOME TO FOODCHAT! Start by describing your dietary needs.';
        document.body.appendChild(popup);
        
        // Remove popup after animation completes
        setTimeout(() => {
            popup.remove();
            style.remove();
        }, 3000);
    }, 500);  // Small delay to ensure Gradio is fully loaded
}
"""