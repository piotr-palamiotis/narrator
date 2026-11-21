"use strict";
const reason = new URLSearchParams(location.search).get("login");
if (reason === "failed") document.getElementById("login-error").hidden = false;
function showNoPassword() {
  document.getElementById("no-password").hidden = false;
  document.getElementById("login-form").hidden = true;
}
if (reason === "nopassword") showNoPassword();
fetch("/api/login-info").then((r) => r.json()).then((info) => {
  if (!info.password_set) showNoPassword();
}).catch(() => {});
