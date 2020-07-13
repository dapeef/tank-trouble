// Global 'constants' set in the onLoad function
var canvas;
var ctx;
var viewRect;
var socket;
var params;

// Key tracking
var pressedKeys = {};

// Variables for the game
var maze = {
    width: 1,
    height: 1,
    walls: []
};
var tanks = {};
var serverTanks = {};

var projectiles = {};
var nextProjectileId = 0;

var lastTime = Date.now();
var wasMovingLastFrame = false;

// Variables used for rendering
var dpr = 1;

/* ===== CONSTANTS ===== */
// Constants that are needed by the physics engine
const TANK_WIDTH = 0.32;
const TANK_LENGTH = 0.42;

// Display constants
const TANK_OUTLINE_THICKNESS = 0.01;

const BARREL_RADIUS = 0.1;
const BARREL_OVERHANG = 0.2;
const TURRET_RADIUS = 0.1;

const MAZE_FILL_FACTOR = 0.95; // What proportion of the canvas should be filled by the maze
const TANK_CAMERA_ZOOM_FACTOR = 0.1; // Proportion of the diagonal length of the canvas window

// Key bindings
const KEY_LEFT = 75;
const KEY_UP = 79;
const KEY_RIGHT = 59;
const KEY_DOWN = 76;

const SHOOT_KEY = 88;

// Gameplay constants
const ROTATION_SPEED = 5; // rad/s
const MOVEMENT_SPEED = 2; // square/s

const MAX_OWNED_BULLETS = 5; // Number of bullets allowed to be owned by one tank at a time

// Debug view settings
var DEBUG_SERVER_TANKS = false;
var DEBUG_RECT_OUTLINES = false;
var DEBUG_RAYCAST = false;

var ATTACH_CAMERA_TO_TANK = false;

// Lag compensation settings
const LATENCY_COMPENSATION_LERP_FACTOR = 12;


// Called when the document loads
function onLoad() {
    // Read the params from the URL
    params = getParams(window.location.href);

    // Get the canvas element and its drawing context
    canvas = document.getElementById("canvas");
    ctx = canvas.getContext("2d");

    // Make sure that the canvas looks good on high DPI monitors (like mine)
    dpr = window.devicePixelRatio || 1;
    viewRect = canvas.getBoundingClientRect();

    canvas.width = viewRect.width * dpr;
    canvas.height = viewRect.height * dpr;

    // Start socketio client
    socket = io.connect('http://' + document.domain + ':' + location.port);

    // Declare useful function
    var overwriteGlobalState = function(state) {
        maze = state.maze;
        tanks = state.tanks;
        serverTanks = state.tanks;
        projectiles = state.projectiles;
    };

    // When the connection is established, tell the server that a new player has arrived
    socket.on('connect', function() {
        socket.emit('c_on_new_user_arrive', {
            colour: '#' + params.colour,
            name: params.name
        });
    });
    socket.on('s_on_new_user_arrive', function(state) {
        if (state.newUserTag == params.name) {
            overwriteGlobalState(state);
        } else {
            tanks[state.newUserTag] = state.tanks[state.newUserTag];
            serverTanks[state.newUserTag] = state.tanks[state.newUserTag];
        }
    });
    socket.on('s_on_user_leave', function(tags) {
        for (var i = 0; i < tags.length; i++) {
            delete tanks[tags[i]];
            delete serverTanks[tag[i]];
        }
    });
    socket.on('s_broadcast', function(state) { updateServerTankState(state); });
    socket.on('s_on_tank_move', function(tankData) {
        var serverTank = serverTanks[tankData.tag];

        // Check that the tank is actually defined, since we might not have recieved the new
        // new tank yet
        if (serverTank) {
            // Copy the fields of the new state into the right tank, since this only sends the updated
            // state, rather than the entire gamestate.  This is one of the rare cases where using TCP
            // is actually an advantage.
            for (field in tankData.newState) {
                serverTank[field] = tankData.newState[field];
            }
        }
    });
    socket.on('s_spawn_projectile', function(newProj) {
        // Check if the projectile is new and isn't one of ours
        if (!(newProj.id in projectiles)) {
            var proj = newProj.projectile;

            // Move the projectile to where it should be
            var timeSinceSpawn = Date.now() - proj.spawnTime;

            proj.x += proj.velX * timeSinceSpawn / 1000;
            proj.y += proj.velY * timeSinceSpawn / 1000;

            // Add the projectile
            projectiles[newProj.id] = proj;
        }
    });
    socket.on('s_on_tank_explode', function(data) {
        var tank = tanks[data.tankTag];

        // Start animation if this tank's explosion is new to us
        if (tank.isAlive) {
            tank.destructionTime = Date.now();
        }

        tank.isAlive = false;

        delete projectiles[data.projectileTag];
    });
    socket.on('s_start_new_game', function(newGameState) {
        overwriteGlobalState(newGameState);
    });

    // Set up callbacks
    window.onkeyup = function(e) { pressedKeys[e.keyCode] = false; };
    window.onkeydown = function(e) {
        pressedKeys[e.keyCode] = true;

        if (e.keyCode == SHOOT_KEY) {
            // Find the number of currently existing bullets owned by this tank by iterating over
            // the IDs, and testing if they start with the current username.
            var numOwnedBullets = 0;
            for (const id in projectiles) {
                if (projectiles[id].type == BULLET_TYPE && id.startsWith(params.name)) {
                    numOwnedBullets += 1;
                }
            }

            var myTank = getMyTank();

            if (numOwnedBullets < MAX_OWNED_BULLETS && myTank && myTank.isAlive) {
                // Calculate the direction and location of the tank barrel
                var dir = new Vec2(Math.cos(myTank.r), Math.sin(myTank.r));

                var newId = params.name + "|" + nextProjectileId;

                projectiles[newId] = spawnBullet(
                    new Vec2(myTank.x, myTank.y).add(dir.mul(TANK_LENGTH * (0.5 + BARREL_OVERHANG))),
                    dir
                );

                socket.emit("c_spawn_projectile", {
                    id: newId,
                    projectile: projectiles[newId]
                });

                nextProjectileId += 1;
            }
        }
    };
    window.onbeforeunload = function() { socket.close(); };
    window.onresize = function() {
        viewRect = canvas.getBoundingClientRect();

        canvas.width = viewRect.width * dpr;
        canvas.height = viewRect.height * dpr;
    };

    // Set the loops going
    setInterval(updateServer, 50);

    frame();
}

// Called slower than the frames so that the server isn't swamped with updates
function updateServer() {
    var myTank = getMyTank();

    if (myTank) {
        socket.emit("c_on_tank_move", {
            tag: params.name,
            newState: {
                x: myTank.x,
                y: myTank.y,
                r: myTank.r
            }
        });
    }
}

// Called once per frame
function frame() {
    /* ===== UPDATES ===== */
    // Calculate the time since last frame for framerate independence
    var timeDelta = (Date.now() - lastTime) / 1000;
    lastTime = Date.now();

    // Control my tank
    var myTank = getMyTank();

    if (myTank && myTank.isAlive) {
        myTank.angularVelocity = 0;
        if (pressedKeys[KEY_LEFT ] == true) { myTank.angularVelocity -= 1; }
        if (pressedKeys[KEY_RIGHT] == true) { myTank.angularVelocity += 1; }

        myTank.forwardVelocity = 0;
        if (pressedKeys[KEY_DOWN] == true) { myTank.forwardVelocity -= 1; }
        if (pressedKeys[KEY_UP  ] == true) { myTank.forwardVelocity += 1; }

        var isMoving = myTank.angularVelocity != 0 || myTank.forwardVelocity != 0;

        if (isMoving || wasMovingLastFrame) {
            // Don't emit to the server, because doing so every frame will overload the server
        }

        wasMovingLastFrame = isMoving;
    }

    // Update all the tanks' positions
    for (const id in tanks) {
        var tank = tanks[id];

        if (id == params.name) {
            // How to update the position of currently controlled tanks if the server tells us
            // something different.  For now do nothing - the client knows best.
        } else {
            // How to update the position of other tanks according to what the server says.
            var sTank = serverTanks[id];

            if (sTank) {
                // Copy across the velocities, so that the tank will do something approximately
                // what the server is saying - but that is smooth regardless.
                tank.angularVelocity = sTank.angularVelocity;
                tank.forwardVelocity = sTank.forwardVelocity;

                tank.x = lerp(tank.x, sTank.x, LATENCY_COMPENSATION_LERP_FACTOR * timeDelta);
                tank.y = lerp(tank.y, sTank.y, LATENCY_COMPENSATION_LERP_FACTOR * timeDelta);
                tank.r = lerp(tank.r, sTank.r, LATENCY_COMPENSATION_LERP_FACTOR * timeDelta);

                // Calculate the distance between where the server thinks the tank should be and
                // where the client thinks the tank should be.  This corresponds to the divergence
                var dX = sTank.x - tank.x;
                var dY = sTank.y - tank.y;
                var d = Math.sqrt(dX * dX + dY * dY);

                /*
                // We compare the cosines of the angles instead of the angles directly, because
                // the cosine function removes the edge case of wrapping angles round the 2pi mark
                if (d > 0.1 || Math.cos(tank.r - sTank.r) < Math.cos(0.3)) {
                    tank.x = sTank.x;
                    tank.y = sTank.y;
                    tank.r = sTank.r;
                }
                */
            }
        }

        var movementStep = tank.forwardVelocity * MOVEMENT_SPEED * timeDelta;

        tank.r += tank.angularVelocity * timeDelta * ROTATION_SPEED;
        tank.x += movementStep * Math.cos(tank.r);
        tank.y += movementStep * Math.sin(tank.r);
    }

    // Update all projectiles
    var projectilesToDestroy = [];

    for (const id in projectiles) {
        if (updateProjectile(projectiles[id], timeDelta)) {
            projectilesToDestroy.push(id);
        }
    }

    while (projectilesToDestroy.length > 0) {
        delete projectiles[projectilesToDestroy.pop()];
    }

    // Detect when my tank is destroyed
    var myTank = getMyTank();

    if (myTank && myTank.isAlive) {
        for (const id in projectiles) {
            var proj = projectiles[id];
            var tankSpaceCoord = inverseTransformCoord(proj, myTank, myTank.r);

            if (Math.abs(tankSpaceCoord.x) <= TANK_LENGTH / 2 + BULLET_RADIUS
             && Math.abs(tankSpaceCoord.y) <= TANK_WIDTH / 2 + BULLET_RADIUS
            ) {
                myTank.isAlive = false;
                myTank.destructionTime = Date.now();

                socket.emit('c_on_tank_explode', { tankTag: params.name, projectileTag: id });

                delete projectiles[id];

                break;
            }
        }
    }

    /* ===== RENDERING ==== */
    // Clear the canvas
    ctx.clearRect(0, 0, viewRect.width, viewRect.height);

    /* Transform the canvas so that the map starts at (0, 0) and one unit corresponds to one
     * square of the maze
     */
    
    // Save the canvas' transformation matrix so that it can be restored at the end of every frame
    ctx.save();

    // Move the origin to the centre of the canvas window
    ctx.translate(viewRect.width / 2, viewRect.height / 2);

    if (ATTACH_CAMERA_TO_TANK && myTank) {
        /* MAKE THE CAMERA FOLLOW THE TANK. */

        // Calculate the length from one corner to the opposite corner of the canvas
        var diagonalLength = Math.sqrt(
            viewRect.width * viewRect.width + viewRect.height * viewRect.height
        );
        
        // Scale all the drawing according to the size of the canvas
        ctx.scale(
            diagonalLength * TANK_CAMERA_ZOOM_FACTOR,
            diagonalLength * TANK_CAMERA_ZOOM_FACTOR
        );

        // Move the camera to be above the tank, with the tank facing upwards
        ctx.rotate(-myTank.r - Math.PI / 2);
        ctx.translate(-myTank.x, -myTank.y);
    } else {
        /* MAKE THE CAMERA STATIC AND MAKE THE MAZE FILL THE WINDOW */
        
        // Find out what scale to use in order to fill the window with the maze
        var scale = Math.min(
            viewRect.width / maze.width,
            viewRect.height / maze.height
        ) * MAZE_FILL_FACTOR;

        // Scale the canvas by the required scale
        ctx.scale(scale * MAZE_FILL_FACTOR, scale * MAZE_FILL_FACTOR);

        // Translate so that the centre of the maze is the centre of the canvas
        ctx.translate(-maze.width / 2, -maze.height / 2);
    }

    // Draw the grid
    ctx.fillStyle = 'black';

    for (var i = 0; i < maze.walls.length; i++) {
        var w = maze.walls[i];

        ctx.fillRect(w.x, w.y, w.width, w.height);
    }

    // Draw the tanks
    for (const id in tanks) {
        drawTank(tanks[id]);
    }

    // Draw projectiles
    ctx.fillStyle = "black";
    for (const id in projectiles) {
        renderProjectile(ctx, projectiles[id]);
    }

    // ===== DEBUG DRAWING =====
    if (DEBUG_SERVER_TANKS) {
        for (const id in serverTanks) {
            drawTank(serverTanks[id], "rgba(0,0,0,0)");
        }
    }

    if (DEBUG_RECT_OUTLINES) {
        var lines = getAllWallBoundingLines(BULLET_RADIUS);

        for (var i = 0; i < lines.length; i++) {
            ctx.beginPath();

            ctx.moveTo(lines[i].p1.x, lines[i].p1.y);
            ctx.lineTo(lines[i].p2.x, lines[i].p2.y);

            ctx.strokeStyle = "red";
            ctx.lineWidth = 0.03;
            ctx.stroke();
        }
    }

    if (DEBUG_RAYCAST) {
        if (myTank) {
            var points = bouncingRaycast(
                new Vec2(myTank.x, myTank.y),
                new Vec2(Math.cos(myTank.r), Math.sin(myTank.r)),
                BULLET_SPEED * BULLET_LIFETIME,
                BULLET_RADIUS
            );

            if (points.length > 0) {
                ctx.beginPath();
                ctx.moveTo(points[0].x, points[0].y);

                for (var i = 1; i < points.length; i++) {
                    ctx.lineTo(points[i].x, points[i].y);
                }

                ctx.lineWidth = 0.01;
                ctx.strokeStyle = myTank.col;
                ctx.stroke();
            }
        }
    }

    // Restore the window to standard transformation
    ctx.restore();

    // Request another frame
    window.requestAnimationFrame(frame);
}





/* ===== STATE-UPDATING CODE ===== */
function updateServerTankState(state) {
    serverTanks = state;
}




/* ===== DRAWING CODE ===== */
// Draw a rectangle with both a fill and a stroke
function fillStrokeRect(x, y, w, h) {
    ctx.fillRect(x, y, w, h);
    ctx.strokeRect(x, y, w, h);
}

// Draw a tank
function drawTank(tank, fillOverride) {
    if (tank.isAlive) {
        drawLiveTank(tank, fillOverride);
    }
}

function drawLiveTank(tank, fillOverride) {
    // Save the canvas and move it so that the tank is at (0, 0) looking upwards
    ctx.save();
    ctx.translate(tank.x, tank.y);
    ctx.rotate(tank.r + Math.PI / 2);

    // Setup the right colours and line widths
    ctx.strokeStyle = "black";
    ctx.fillStyle = fillOverride || tank.col;
    ctx.lineWidth = 0.01;

    // Tank body
    fillStrokeRect(
        TANK_WIDTH * -0.5,
        TANK_LENGTH * -0.5,
        TANK_WIDTH,
        TANK_LENGTH
    );

    // Barrel
    fillStrokeRect(
        TANK_WIDTH * -BARREL_RADIUS,
        TANK_LENGTH * -(0.5 + BARREL_OVERHANG),
        TANK_WIDTH * BARREL_RADIUS * 2,
        TANK_LENGTH * (0.5 + BARREL_OVERHANG)
    );

    // Turret
    ctx.beginPath();
    ctx.arc(0, 0, TURRET_RADIUS, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();

    // Reset the canvas to where it was before drawing the tank
    ctx.restore();
}





/* ===== COLLISION ENGINE CODE ===== */
function transformCoord(coord, origin, rotation) {
    var rotatedX = coord.x * Math.cos(rotation) - coord.y * Math.sin(rotation);
    var rotatedY = coord.x * Math.sin(rotation) + coord.y * Math.cos(rotation);

    return {
        x: rotatedX + origin.x,
        y: rotatedY + origin.y
    };
}

function inverseTransformCoord(coord, origin, rotation) {
    var translatedX = coord.x - origin.x;
    var translatedY = coord.y - origin.y;

    return {
        x: translatedX * Math.cos(-rotation) - translatedY * Math.sin(-rotation),
        y: translatedX * Math.sin(-rotation) + translatedY * Math.cos(-rotation)
    };
}




/* ===== UTILITIES ===== */
function lerp(a, b, t) {
    return (1 - t) * a + t * b;
}

// Returns t such that lerp(a, b, t) = c.  Divides by 0 if a = b
function inverseLerp(a, b, c) {
    return (c - a) / (b - a);
}

function getMyTank() {
    return tanks[params.name];
}

/**
 * Get the URL parameters
 * source: https://css-tricks.com/snippets/javascript/get-url-variables/
 * @param  {String} url The URL
 * @return {Object}     The URL parameters
 */
function getParams(url) {
	var params = {};
	var parser = document.createElement('a');
	parser.href = url;
	var query = parser.search.substring(1);
	var vars = query.split('&');
	for (var i = 0; i < vars.length; i++) {
		var pair = vars[i].split('=');
		params[pair[0]] = decodeURIComponent(pair[1]);
	}
	return params;
};
