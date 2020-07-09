// Global constants
var HUDcontainer;
var HUDreadouts;

function redrawHUD() {
    for (let i = 0; i < HUDreadouts.length; i++) {
        const readout = HUDreadouts[i];

        readout.parentNode.removeChild(readout);
    }

    HUDreadouts = [];

    HUDcontainer.style.gridTemplateColumns = "auto " * tanks.length

    for (let i = 0; i < tanks.length; i++) {
        const tank = tanks[i];

        HUDreadouts.push(document.createElement("div"));

        HUDcontainer.appendChild(HUDreadouts[HUDreadouts.length])
    }
}