var ImgCanva;
var DrawCanva;
var SelectedAreaCanva;
var canva_selectedImg;
var img_all;
var grid;
document.addEventListener("DOMContentLoaded", function() {
    ImgCanva = document.getElementById("canva_allImg").getContext("2d", { willReadFrequently: true });
    DrawCanva = document.getElementById("canva_drawing").getContext("2d");
    img_all = new Image();
    img_all.onload = () => {
        ImgCanva.drawImage(img_all, 0, 0);
        // ADD WIDTH AND HEIGHT TO THE JSON
        COLOR_LABEL.set("width", img_all.width);
        COLOR_LABEL.set("height", img_all.height);
        COLOR_LABEL.set("area_count", 0);
    };
    img_all.src = imageUrl;

    canva_selectedImg = document.getElementById("canva_selectedImg");
    SelectedAreaCanva = canva_selectedImg.getContext("2d");

    grid = document.getElementById("grid");

    //SCALE NAVBAR
    scaleNavBar();

    //SET COLORFILTER (RGB & HSV & INTERVAL) VALUES
    updateColor(document.getElementById("colorSelectorInput"));
    interval = Number.parseInt(document.getElementById("intervalSelectorInupt").value);

    //LOAD JSON IF PRESENT
    if (jsonUrl != "") {
        fetch(jsonUrl).then((response) => response.json()).then((data) => restoreJSON(data));
    }
});

function restoreJSON(data) {
    for (const prop in data) {
        if (prop[0] == 'R') {
            var map = new Map();
            for (const subprop in data[prop]) {
                var obj = JSON.parse(data[prop][subprop]);
                let xmin, xmax, ymin, ymax;
                for (const p in obj) {
                    const x = obj[p][0];
                    const y = obj[p][1];
                    DrawCanva.fillRect(x, y, 1, 1);

                    if (xmin == null || xmin == undefined) {
                        xmin = x;
                        xmax = x;
                        ymin = y;
                        ymax = y;
                    } else {
                        if (x < xmin)
                            xmin = x;
                        if (x > xmax)
                            xmax = x;
                        if (y < ymin)
                            ymin = y;
                        if (y > ymax)
                            ymax = y;
                    }
                }
                DrawCanva.beginPath();
                DrawCanva.rect(xmin - 5, ymin - 5, xmax - xmin + 10, ymax - ymin + 10);
                DrawCanva.stroke();
                map.set(subprop, JSON.stringify(obj));
            }
            COLOR_LABEL.set(prop, Object.fromEntries(map))
        }
        if (prop == "area_count") {
            selectedArea_count = data[prop];
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////
//   ALL THE FUNCTIONS TO DRAW THE RECTANGLE/AREA OF SELECTION IN THE INPUT IMAGE
/////////////////////////////////////////////////////////////////////////////////
var Rectangle = /** @class */ (function() {
    function Rectangle(posX, posY) {
        this.x = posX;
        this.y = posY;
        this.w = 0;
        this.h = 0;
    }
    Rectangle.prototype.setSurface = function(width, height) {
        this.w = width - this.x;
        this.h = height - this.y;
    };

    // Use to debug
    Rectangle.prototype.toString = function() {
        return this.x + " : " + this.y + " : " + this.w + " : " + this.h;
    };

    // Use to make sure that the top left corner of the rectangle is at [x,y] with the good width and height
    // Update ocure when the user do not draw the rectangle from the top left corner to the bottom right. 
    Rectangle.prototype.updateCoord = function() {
        if (this.w < 0) {
            this.x = this.x + this.w;
            this.w = this.w * -1;
        }
        if (this.h < 0) {
            this.y = this.y + this.h;
            this.h = this.h * -1;
        }
    };
    return Rectangle;
}());

var rect;
var isDrawing, hasMove;

function canvaDrawing_mouseDown(e) {
    rect = new Rectangle(e.offsetX, e.offsetY);
    isDrawing = true;
    hasMove = false;
}

function canvaDrawing_mouseMove(e) {
    if (isDrawing) {
        // CLEAN THE CANVAS
        if (rect.w >= 0 && rect.h >= 0)
            DrawCanva.clearRect(rect.x - 1, rect.y - 1, rect.w + 2, rect.h + 2);
        if (rect.w < 0 && rect.h < 0)
            DrawCanva.clearRect(rect.x + 1, rect.y + 1, rect.w - 2, rect.h - 2);
        if (rect.w < 0)
            DrawCanva.clearRect(rect.x + 1, rect.y - 1, rect.w - 2, rect.h + 2);
        if (rect.h < 0)
            DrawCanva.clearRect(rect.x - 1, rect.y + 1, rect.w + 2, rect.h - 2);
        // UPDATE THE RECTANGLE OBJECT
        rect.setSurface(e.offsetX, e.offsetY);
        // DRAW THE NEW RECTANGLE ON THE CANVAS
        DrawCanva.beginPath();
        DrawCanva.rect(rect.x, rect.y, rect.w, rect.h);
        DrawCanva.stroke();
        // SET MOVEMENT TRUE (PREVENT CLICK EVENT)
        hasMove = true;
    }
}

function canvaDrawing_mouseUp() {
    // END MOVEMENT
    isDrawing = false;
    hasMove = false;

    // CLEAN SELECTED PIXELS ARRAY
    while (selectedPixels.length > 0) { selectedPixels.pop(); }

    //UPDATE THE RECTANGLE
    rect.updateCoord();

    // UPDATE ZOOM CANVAS & GRID
    updateZoomArea();

    // ENABLED THE USER TO DRAW IN THE GRID
    document.getElementById("draw_button").disabled = false;
}

function updateZoomArea() {
    canva_selectedImg.width = rect.w;
    canva_selectedImg.height = rect.h;
    SelectedAreaCanva.drawImage(img_all, rect.x, rect.y, rect.w, rect.h, 0, 0, rect.w, rect.h);

    grid.style.width = canva_selectedImg.offsetWidth + "px";
    grid.style.height = canva_selectedImg.offsetHeight + "px";
    grid.innerHTML = "";
    grid.style.gridTemplateColumns = "repeat(" + rect.w + ",1fr);";
    grid.style.gridTemplateRows = "repeat(" + rect.h + ",1fr);";
    for (var i = 1; i < rect.h; i++) {
        for (var j = 1; j < rect.w; j++) {
            var grid_el = document.createElement("div");
            grid_el.style.gridArea = i + " / " + j + " / " + (i + 1) + " / " + (j + 1);
            grid_el.setAttribute('x', j - 1);
            grid_el.setAttribute('y', i - 1);
            grid_el.classList.add("grid_el");
            // EVENT HANDLER
            grid_el.addEventListener("mouseenter", grid_mouseenter);
            grid_el.addEventListener("mousedown", grid_mousedown);
            grid_el.addEventListener("mouseup", grid_mouseup);
            grid.appendChild(grid_el);
        }
    }
}

var selectedPixels = Array();
var grid_el_hover;
var drawInGrid = false;

function grid_mousedown(event) {
    drawInGrid = true;
    if (draw_mode && selectedPixels.indexOf(event.target) == -1) {
        selectedPixels.push(event.target);
        event.target.classList.add("labelize");
        document.getElementById("delete_button").disabled = false;
        document.getElementById("areaOK_button").disabled = false;
    }
    if (delete_mode && selectedPixels.indexOf(event.target) != -1) {
        var index = selectedPixels.indexOf(event.target);
        selectedPixels = selectedPixels.filter(function(val, i) { return i != index; });
        event.target.classList.remove("labelize");
        if (selectedPixels.length == 0)
            document.getElementById("areaOK_button").disabled = true;
    }
}

function grid_mouseenter(event) {
    grid_el_hover = event.target;
    if (draw_mode && drawInGrid) {
        if (grid_el_hover.style.backgroundColor != "red" && selectedPixels.indexOf(grid_el_hover) == -1) {
            selectedPixels.push(grid_el_hover);
            grid_el_hover.classList.add("labelize");
            document.getElementById("delete_button").disabled = false;
            document.getElementById("areaOK_button").disabled = false;
        }
    }
    if (delete_mode && drawInGrid) {
        if (grid_el_hover.className == "grid_el labelize" && selectedPixels.indexOf(grid_el_hover) != -1) {
            var index = selectedPixels.indexOf(grid_el_hover);
            selectedPixels = selectedPixels.filter(function(val, i) { return i != index; });
            grid_el_hover.classList.remove("labelize");
            if (selectedPixels.length == 0)
                document.getElementById("areaOK_button").disabled = true;
        }
    }
}

function grid_mouseup(event) {
    drawInGrid = false;
}

// MODE OF PIXELS SELECTION IN THE GRID
var draw_mode = true;
var delete_mode = false;

function draw_button_click() {
    draw_mode = true;
    document.getElementById("draw_button").classList.add("activeBtn");
    delete_mode = false;
    document.getElementById("delete_button").classList.remove("activeBtn");
}

function delete_button_click(e) {
    draw_mode = false;
    document.getElementById("draw_button").classList.remove("activeBtn");
    delete_mode = true;
    e.target.classList.add("activeBtn");
}

var COLOR_LABEL = new Map();
var AREAS = new Map();
var selectedArea_count = 0;

function getCookie(name) {
    let cookieValue = null;
    if (document.cookie && document.cookie !== '') {
        const cookies = document.cookie.split(';');
        for (let i = 0; i < cookies.length; i++) {
            const cookie = cookies[i].trim();
            // Does this cookie string begin with the name we want?
            if (cookie.substring(0, name.length + 1) === (name + '=')) {
                cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                break;
            }
        }
    }
    return cookieValue;
}

function areaOK_click(e) {
    // SET "areaX" IN JSON 
    var pixels = [];
    selectedPixels.forEach(element => {
        var x = Number.parseInt(element.getAttribute('x')) + rect.x;
        var y = Number.parseInt(element.getAttribute('y')) + rect.y;
        pixels.push(new Array(x, y));
        DrawCanva.fillRect(x, y, 1, 1);
    });
    while (selectedPixels.length > 0) { selectedPixels.pop(); }
    AREAS.set("area" + selectedArea_count, JSON.stringify(pixels))

    // SET AREA COUNT
    selectedArea_count++;
    COLOR_LABEL.set("area_count", selectedArea_count);

    // SET RGB-X-X-X WITH ALL AREAS
    let key = "RGB" + color.rgb.r + "-" + color.rgb.g + "-" + color.rgb.b;
    checkforPrevLabel(key);
    COLOR_LABEL.set(key, Object.fromEntries(AREAS));

    // CREATE JSON FROM PREVIOUS OBJECTS (using form)
    var json = JSON.stringify(Object.fromEntries(COLOR_LABEL));
    var file = new File([json], imageName.split('.')[0] + '.json');
    var url = '/labeler/' + imageID + '/';
    var csrftoken = getCookie('csrftoken');
    var formData = new FormData();
    formData.append("jsonFile", file);
    formData.append("workingStatus", false);

    // SEND THE JSON
    var xhr = new XMLHttpRequest();
    xhr.open("POST", url, true);
    xhr.setRequestHeader('X-CSRFToken', csrftoken);
    xhr.setRequestHeader('enctype', "multipart/form-data")
    xhr.send(formData);

    // UPDATE BUTTON AND GRID
    document.getElementById("areaOK_button").setAttribute("disabled", true);
    grid.childNodes.forEach(child => grid.removeChild(child));
    grid.innerHTML = "Selectionner une nouvelle zone sur l'image";
    SelectedAreaCanva.clearRect(0, 0, rect.w, rect.h);

}

function checkforPrevLabel(key) {
    if (COLOR_LABEL.has(key)) {
        for (const areas in COLOR_LABEL.get(key)) {
            AREAS.set(areas, COLOR_LABEL.get(key)[areas]);
        }
    }
}


function FileListItems(file) {
    var b = new ClipboardEvent("").clipboardData || new DataTransfer()
    b.items.add(file)
    return b.files
}

function wholeImgOk_click(e) {
    if (!window.confirm(selectedArea_count > 0 ? selectedArea_count + " pigment(s) ont été labélisé" : "Aucun pigment sur cette image?")) { return; }

    updateJSON();
    var json = JSON.stringify(Object.fromEntries(COLOR_LABEL));
    var file = new File([json], imageName.split('.')[0] + '.json');
    var input = document.getElementById("jsonFile");
    var button = document.getElementById("jsonButton");
    var checkbox = document.getElementById("workingStatus")



    input.files = new FileListItems(file);
    checkbox.checked = true;
    button.click();

}

function scaleNavBar() {
    var width = document.getElementById("canva_allImg").getAttribute("width");
    width = (width < window.screen.width) ? window.screen.width : width;
    document.getElementById("navbar").style.width = width + "px";
}

var filter = false;

function filter_click(element) {
    document.getElementById("filterBtn").classList.toggle("activeBtn");
    filter = !filter;
    if (!filter) {
        ImgCanva.drawImage(img_all, 0, 0);
        element.innerHTML = "Activer";
    } else {
        applyFilter();
        element.innerHTML = "Désactiver";
    }
}

function applyFilter() {
    if (!filter) { return; }
    var idata = ImgCanva.getImageData(0, 0, img_all.width, img_all.height);
    var data = idata.data;
    var newData = new ImageData(img_all.width, img_all.height);

    for (var i = 0; i < data.length; i += 4) {
        let hsv = rgb2hsv(data[i], data[i + 1], data[i + 2]);

        newData.data[i] = data[i];
        newData.data[i + 1] = data[i + 1];
        newData.data[i + 2] = data[i + 2];
        if (hsv['h'] >= color.hsv.h - interval && hsv['h'] <= color.hsv.h + interval) {
            newData.data[i + 3] = 255;
        } else {
            newData.data[i + 3] = 20;
        }
    }
    ImgCanva.putImageData(newData, 0, 0);
}

function updateJSON() {
    if (AREAS.size > 0) {
        COLOR_LABEL.set("RGB" + color.rgb.r + "-" + color.rgb.g + "-" + color.rgb.b, Object.fromEntries(AREAS));
        AREAS.clear();
    }
}

var color = {
    hsv: { h: 0, s: 0, v: 0 },
    rgb: { r: 0, g: 0, b: 0 }
}

function updateColor(element) {
    var val = element.value;
    color.rgb.r = parseInt(val.substr(1, 2), 16)
    color.rgb.g = parseInt(val.substr(3, 2), 16)
    color.rgb.b = parseInt(val.substr(5, 2), 16)

    var hsv = rgb2hsv(color.rgb.r, color.rgb.g, color.rgb.b);
    color.hsv.h = hsv.h;
    color.hsv.s = hsv.s;
    color.hsv.v = hsv.v;
}

function colorInput_onChange(element) {
    updateJSON();
    updateColor(element);
}

/*
From https://stackoverflow.com/questions/8022885/rgb-to-hsv-color-in-javascript by Mic and aetonsi
*/
function rgb2hsv(r, g, b) {
    r = r / 255;
    g = g / 255;
    b = b / 255;
    let v = Math.max(r, g, b),
        c = v - Math.min(r, g, b);
    let h =
        c && (v == r ? (g - b) / c : v == g ? 2 + (b - r) / c : 4 + (r - g) / c);
    return {
        h: (60 * (h < 0 ? h + 6 : h)),
        s: 255 * (v && c / v),
        v: 255 * v,
    };
}

var interval;

function intervalSelectorInupt_onChange(element) {
    interval = Number.parseInt(element.value);
    if (filter) {
        applyFilter();
    }
}