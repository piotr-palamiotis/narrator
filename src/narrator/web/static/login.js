"use strict";
if (new URLSearchParams(location.search).get("login") === "failed") {
  document.getElementById("login-error").hidden = false;
}
