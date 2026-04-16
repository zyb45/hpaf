#!/usr/bin/env bash
set -Eeuo pipefail

########################################
# 基本路径配置
########################################
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

ROS_SETUP="/opt/ros/humble/setup.bash"
CONDA_SH="$HOME/miniforge3/etc/profile.d/conda.sh"

CAM_SCRIPT="$HOME/piper_learning/piper_sdk-master/piper_sdk/can_activate.sh"
GAMEPAD_DIR="$HOME/piper_learning/Gamepad_PiPER"

HPAF_SHARED_ROOT="$ROOT_DIR/shared_scene"
LOG_DIR="$ROOT_DIR/logs/startup"

# rviz2 配置文件：没有的话就直接启动空白 rviz2
RVIZ_CONFIG="${RVIZ_CONFIG:-$ROOT_DIR/rviz/hpaf_dual_camera.rviz}"

########################################
# 运行参数
########################################
USE_TERMINATOR=1        # 1: 优先用 terminator 开 tab；0: 用 gnome-terminal
ENABLE_RVIZ=1           # 1: 启动 rviz2；0: 不启动
GAMEPAD_USE_SUDO=1      # 1: 用 sudo 启动 ctrl_human.py；0: 普通方式启动
PRECACHE_SUDO=1         # 1: 启动前先 sudo -v
KEEP_SUDO_ALIVE=1       # 1: 后台维持 sudo 凭据

########################################
# 公共环境
########################################
mkdir -p "$HPAF_SHARED_ROOT/primary" "$HPAF_SHARED_ROOT/secondary" "$LOG_DIR"

COMMON_PREFIX="source '$ROS_SETUP'"
CONDA_PREFIX_CMD="source '$CONDA_SH'"

########################################
# 小工具函数
########################################
info()  { echo -e "\033[1;32m[INFO]\033[0m $*"; }
warn()  { echo -e "\033[1;33m[WARN]\033[0m $*"; }
error() { echo -e "\033[1;31m[ERR ]\033[0m $*"; }

cleanup() {
  if [[ -n "${SUDO_KEEPALIVE_PID:-}" ]]; then
    kill "$SUDO_KEEPALIVE_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    error "缺少命令: $1"
    exit 1
  }
}

run_in_term() {
  local title="$1"
  local cmd="$2"

  if [[ "$USE_TERMINATOR" -eq 1 ]] && command -v terminator >/dev/null 2>&1; then
    terminator --new-tab --title="$title" -x bash -lc "$cmd; exec bash" >/dev/null 2>&1 &
    sleep 0.3
  elif command -v gnome-terminal >/dev/null 2>&1; then
    gnome-terminal --tab --title="$title" -- bash -lc "$cmd; exec bash" >/dev/null 2>&1 &
    sleep 0.3
  else
    error "既没有 terminator 也没有 gnome-terminal，无法自动开终端。"
    exit 1
  fi
}

########################################
# 启动前检查
########################################
need_cmd bash
need_cmd tee
need_cmd ros2
need_cmd python3

if [[ "$USE_TERMINATOR" -eq 1 ]]; then
  if ! command -v terminator >/dev/null 2>&1; then
    warn "未检测到 terminator，自动回退到 gnome-terminal"
    USE_TERMINATOR=0
  fi
fi

if [[ "$USE_TERMINATOR" -eq 0 ]]; then
  need_cmd gnome-terminal
fi

########################################
# sudo 预认证
########################################
if [[ "$PRECACHE_SUDO" -eq 1 ]]; then
  info "准备预缓存 sudo 凭据，请在当前终端输入一次密码。"
  sudo -v

  if [[ "$KEEP_SUDO_ALIVE" -eq 1 ]]; then
    info "后台维持 sudo 凭据有效。"
    (
      while true; do
        sudo -n true
        sleep 50
      done
    ) &
    SUDO_KEEPALIVE_PID=$!
  fi
fi

########################################
# 启动说明
########################################
cat <<EOF
将要启动：
1) Gemini 相机
2) Gemini 深度镜像修正 service
3) Astra 相机
4) 双相机写盘 ros_bridge
5) CAN 激活
6) Gamepad_PiPER 手动控制
7) RViz2（可选）
8) HPAF 项目终端

日志目录：
$LOG_DIR

共享目录：
$HPAF_SHARED_ROOT
EOF

########################################
# 各任务命令
########################################

# 1. 先启动 Gemini
CMD_GEMINI="
$COMMON_PREFIX
ros2 launch orbbec_camera gemini.launch.xml \
  2>&1 | tee '$LOG_DIR/gemini.log'
"

# 2. 启动 Astra
CMD_ASTRA="
$COMMON_PREFIX
ros2 launch orbbec_camera astra_pro_plus.launch.xml \
  2>&1 | tee '$LOG_DIR/astra.log'
"

# 3. Gemini 镜像修正 + LDP 关闭
CMD_GEMINI_MIRROR_FIX="
$COMMON_PREFIX
echo '[INFO] 等待 /camera_gemini/set_depth_mirror 服务出现...'
for i in {1..25}; do
  if ros2 service list | grep -Fxq '/camera_gemini/set_depth_mirror'; then
    echo '[INFO] 服务已出现，开始调用 set_depth_mirror=false'
    ros2 service call /camera_gemini/set_depth_mirror std_srvs/srv/SetBool '{data: false}' \
      2>&1 | tee '$LOG_DIR/gemini_mirror.log'
    
    echo '[INFO] 等待 /camera_gemini/set_ldp_enable 服务出现...'
    for j in {1..10}; do
      if ros2 service list | grep -Fxq '/camera_gemini/set_ldp_enable'; then
        echo '[INFO] LDP 服务已出现，开始调用 set_ldp_enable=false'
        ros2 service call /camera_gemini/set_ldp_enable std_srvs/srv/SetBool '{data: false}' \
          2>&1 | tee -a '$LOG_DIR/gemini_mirror.log'
        exit 0
      fi
      sleep 1
    done
    echo '[WARN] 等待 /camera_gemini/set_ldp_enable 超时，跳过 LDP 关闭' | tee -a '$LOG_DIR/gemini_mirror.log'
    exit 0
  fi
  sleep 1
done
echo '[ERR] 等待 /camera_gemini/set_depth_mirror 超时' | tee '$LOG_DIR/gemini_mirror.log'
"

# 4. 双相机写盘
CMD_DUAL_DUMP="
$COMMON_PREFIX
cd '$ROOT_DIR'
export HPAF_SHARED_ROOT='$HPAF_SHARED_ROOT'
export HPAF_PRIMARY_NS='/camera_gemini'
export HPAF_SECONDARY_NS='/camera_astra'
python3 ros_bridge/ros2_dual_camera_dump.py \
  2>&1 | tee '$LOG_DIR/dual_dump.log'
"

# 5. CAN 激活
CMD_CAN="
cd '$(dirname "$CAM_SCRIPT")'
bash '$CAM_SCRIPT' can0 1000000 \
  2>&1 | tee '$LOG_DIR/can.log'
"

# 6. 手动控制
if [[ "$GAMEPAD_USE_SUDO" -eq 1 ]]; then
  CMD_GAMEPAD="
$CONDA_PREFIX_CMD
conda activate test_tracik
cd '$GAMEPAD_DIR'
sudo -E env PATH=\"\$PATH\" bash -lc '
  source \"$CONDA_SH\"
  conda activate test_tracik
  cd \"$GAMEPAD_DIR\"
  python ctrl_human.py
' 2>&1 | tee '$LOG_DIR/gamepad.log'
"
else
  CMD_GAMEPAD="
$CONDA_PREFIX_CMD
conda activate test_tracik
cd '$GAMEPAD_DIR'
python ctrl_human.py \
  2>&1 | tee '$LOG_DIR/gamepad.log'
"
fi

# 7. HPAF 项目终端
CMD_PROJECT="
$CONDA_PREFIX_CMD
conda activate piper
cd '$ROOT_DIR'
export PYTHONPATH='$ROOT_DIR'
echo 'HPAF 项目终端已就绪。共享目录在 $HPAF_SHARED_ROOT'
exec bash
"

# 8. RViz2
if [[ -f "$RVIZ_CONFIG" ]]; then
  CMD_RVIZ="
$COMMON_PREFIX
sleep 10
rviz2 -d '$RVIZ_CONFIG' \
  2>&1 | tee '$LOG_DIR/rviz2.log'
"
else
  CMD_RVIZ="
$COMMON_PREFIX
sleep 10
echo '[WARN] 未找到 rviz 配置文件: $RVIZ_CONFIG'
rviz2 \
  2>&1 | tee '$LOG_DIR/rviz2.log'
"
fi

########################################
# 启动顺序
########################################
info "开始按顺序拉起各模块..."

# 先起 Gemini
run_in_term "HPAF-Gemini" "$CMD_GEMINI"

# 等 Gemini 初始化一会儿，再修正深度镜像
sleep 3
run_in_term "HPAF-Gemini-MirrorFix" "$CMD_GEMINI_MIRROR_FIX"

# 再起 Astra
sleep 2
run_in_term "HPAF-Astra" "$CMD_ASTRA"

# 再起双相机写盘
sleep 2
run_in_term "HPAF-DualDump" "$CMD_DUAL_DUMP"

# 机械臂 CAN
sleep 1
run_in_term "HPAF-CAN" "$CMD_CAN"

# 手柄控制
sleep 1
run_in_term "HPAF-Gamepad" "$CMD_GAMEPAD"

# RViz2
if [[ "$ENABLE_RVIZ" -eq 1 ]]; then
  sleep 1
  run_in_term "HPAF-RViz2" "$CMD_RVIZ"
fi

# 项目终端
sleep 1
run_in_term "HPAF-Project" "$CMD_PROJECT"

info "全部启动命令已发出。"
info "当前顺序：Gemini -> Gemini镜像修正 -> Astra -> DualDump -> CAN -> Gamepad -> RViz2 -> Project"
