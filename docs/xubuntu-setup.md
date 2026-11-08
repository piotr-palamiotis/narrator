# Setting up a Xubuntu machine for narrator + OBS

This guide prepares a dedicated Xubuntu machine to run narrator, OBS, and a
fullscreen OBS output (projector) on a chosen screen, with remote access.

It targets **Xubuntu 24.04 LTS** or a newer Xubuntu LTS. Xubuntu runs the
Xfce desktop on **X11**. X11 matters here: remote-control software,
placing a fullscreen window on a specific screen, and turning off screen
blanking all work without the permission prompts Wayland uses.
(Ubuntu's default GNOME desktop no longer has an X11 session from 25.10
on.)

> Status: written 2026-10-17. Steps marked *(verify)* have not been run on
> a Xubuntu machine yet. Check them on the first install and correct this
> guide.

Commands are typed in a terminal (Applications → Terminal Emulator). Lines
starting with `sudo` ask for your password.

---

## 1. Install Xubuntu

1. Download the Xubuntu LTS image from https://xubuntu.org/download/ and
   write it to a USB stick (for example with balenaEtcher or
   `Startup Disk Creator`).
2. Boot from the stick and choose **Install Xubuntu**.
3. Choose the **minimal installation**. Do not tick "Install third-party
   software for graphics" here; the NVIDIA driver is installed in step 3.
4. Create the user that will run narrator (in this guide: `narrator`).
   Choose **Log in automatically**, or set autologin up later (step 2).
5. After the first boot, update the system:

   ```bash
   sudo apt update && sudo apt full-upgrade -y
   sudo reboot
   ```

## 2. Log in automatically into the X11 session

The machine must reach the desktop without anyone at the keyboard. OBS
needs a desktop session to start.

Create the file `/etc/lightdm/lightdm.conf.d/50-autologin.conf`:

```bash
sudo mkdir -p /etc/lightdm/lightdm.conf.d
sudo tee /etc/lightdm/lightdm.conf.d/50-autologin.conf > /dev/null <<'EOF'
[Seat:*]
autologin-user=narrator
autologin-user-timeout=0
autologin-session=xubuntu
EOF
```

Replace `narrator` with your user name. Reboot and check that the desktop
appears without a login screen.

Check that the session is X11:

```bash
echo $XDG_SESSION_TYPE      # must print: x11
```

## 3. NVIDIA driver

```bash
ubuntu-drivers devices          # lists the recommended driver
sudo ubuntu-drivers install     # installs the recommended one
sudo reboot
```

After the reboot:

```bash
nvidia-smi                      # shows the card, driver version and usage
sudo apt install -y libvulkan1  # Vulkan loader (narrator uses Vulkan today)
```

Optional check: `sudo apt install -y vulkan-tools && vulkaninfo --summary`
should list the NVIDIA card. If the machine also has an Intel iGPU, it is
listed too; narrator picks the discrete card automatically.

## 4. Never blank, lock or sleep the screens

The projector screen must stay on at all times.

1. **Power manager**: Settings → Power Manager → Display tab: turn
   "Display power management" off, and set "Blank after" to Never. On the
   System tab, set sleep to Never. The same from a terminal *(verify)*:

   ```bash
   xfconf-query -c xfce4-power-manager -p /xfce4-power-manager/dpms-enabled -n -t bool -s false
   xfconf-query -c xfce4-power-manager -p /xfce4-power-manager/blank-on-ac -n -t int -s 0
   xfconf-query -c xfce4-power-manager -p /xfce4-power-manager/dpms-on-ac-sleep -n -t int -s 0
   xfconf-query -c xfce4-power-manager -p /xfce4-power-manager/dpms-on-ac-off -n -t int -s 0
   ```

2. **Screensaver and lock**: Settings → Screensaver: turn the screensaver
   and "Lock Screen" off. From a terminal *(verify)*:

   ```bash
   xfconf-query -c xfce4-screensaver -p /saver/enabled -n -t bool -s false
   xfconf-query -c xfce4-screensaver -p /lock/enabled -n -t bool -s false
   ```

   If `light-locker` is installed (older Xubuntu releases), remove it:
   `sudo apt remove -y light-locker`.

3. **X11 blanking** as a backstop: Settings → Session and Startup →
   Application Autostart → Add. Name it "No blanking", command:

   ```bash
   sh -c "xset s off -dpms s noblank"
   ```

4. **Machine sleep**: make sure the machine itself never suspends:

   ```bash
   sudo systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target
   ```

## 5. Screens

1. Connect all screens, then open Settings → Display. Arrange them, set
   the resolution and refresh rate, and press **Apply**. Xfce remembers the
   layout for this set of screens.
2. Write down the output names. narrator will store the projector screen
   by this name, so it survives screens being plugged in a different order:

   ```bash
   xrandr --query | grep " connected"
   # example: HDMI-0 connected 1920x1080+1920+0 ...
   ```

3. Hide the mouse cursor when it doesn't move, so it never appears on the
   projector:

   ```bash
   sudo apt install -y unclutter-xfixes
   ```

   Add it to Application Autostart (as in step 4.3) with the command
   `unclutter --timeout 2`.

## 6. Notifications and updates that could cover the output

Pop-ups must never appear on top of the projector.

- Settings → Notifications: turn on **Do not disturb**.
- Update pop-ups: Settings → Software & Updates → Updates tab: set
  "Notify me of a new Ubuntu version" to **Never**, and "When there are
  security updates" to "Download and install automatically".
- Automatic reboots after updates stay **off** (the default). Do not set
  `Unattended-Upgrade::Automatic-Reboot "true"`. Plan reboots yourself,
  outside events.

## 7. Remote access

Install your remote-control software as its vendor describes for Ubuntu
(.deb package). On X11 it can see and control the whole desktop without a
confirmation prompt.

- Turn on its **unattended access** (a fixed password or account login),
  so it works after a reboot with nobody at the machine.
- Check that it starts at boot before the first event: reboot and connect
  remotely without touching the machine.

For SSH access (also used to reach narrator's web UI safely):

```bash
sudo apt install -y openssh-server
```

narrator's web UI listens only on the machine itself (127.0.0.1:8765).
From another computer, open it through an SSH tunnel:

```bash
ssh -L 8765:127.0.0.1:8765 narrator@<machine-address>
# then open http://127.0.0.1:8765 in your browser
```

## 8. Install narrator and OBS

```bash
sudo apt install -y git make
git clone <narrator repository URL> ~/narrator
cd ~/narrator
make setup
```

`make setup` installs the system packages, checks the NDI runtime, builds
the speech recognition with GPU support, picks a performance preset,
installs OBS + DistroAV (the NDI plugin), installs narrator as a user
service and runs a smoke check. It prints what it does and asks before
changing the system (`YES=1` skips the questions, `DRY_RUN=1` only prints
the commands).

The NDI runtime is a separate install from https://ndi.video;
`make setup` tells you if it is missing.

Check the service:

```bash
make status       # is narrator running?
make logs         # live log (Ctrl+C to leave)
```

## 9. OBS fullscreen output (planned, todo C9)

Planned: narrator starts and supervises OBS itself, turns off the preview
in the OBS main window, and opens a fullscreen **Program projector** on the
screen chosen in narrator's web UI. Until that is built, do it by hand:

1. In OBS, right-click the preview → untick **Enable Preview**.
2. Right-click the preview area (or the Program view in Studio Mode) →
   **Fullscreen Projector (Program)** → choose the screen.
3. Settings → General → Projectors: tick **Save projectors on exit**, so
   OBS reopens the projector at its next start.
4. Settings → General: tick **Hide OBS windows from screen capture** if your
   remote software should not show OBS twice *(optional)*.
5. Start OBS automatically: Settings (Xfce) → Session and Startup →
   Application Autostart → Add, command:

   ```bash
   obs --disable-shutdown-check --minimize-to-tray
   ```

## 10. Final check after a reboot

Reboot, don't touch the machine, and check:

- [ ] The desktop appears without a login screen (X11 session).
- [ ] `nvidia-smi` shows the card.
- [ ] narrator is running (`make status`), and its web UI opens through the
      SSH tunnel.
- [ ] OBS has started and the projector is fullscreen on the right screen.
- [ ] Remote-control software connects without anyone at the machine.
- [ ] After 30 minutes with no input, no screen is blank or locked.
