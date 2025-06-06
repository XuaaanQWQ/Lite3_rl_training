#!/usr/bin/env bash
# scripts/auto_train.sh
#
# bash scripts/auto_train.sh

### --------- 超参列表 ------------
# 格式：<参数名>=<新值>
PARAM_SETS=(
  "stand_still=-0.2"
  "ang_vel_xy=-0.5 base_height=-1.6 dof_vel=-0.0001 feet_air_time=1.2 orientation=-5.5 stand_still=-0.7 tracking_ang_vel=1.2 tracking_lin_vel=2.5 dof_acc=-1e-7" 
)
### --------------------------------

PY_FILE="../envs/lite3/lite3_config.py" 
BACKUP="${PY_FILE}.bak$$"  
RUN_CMD="python3 ./train.py \
  --rl_device cuda:0 --sim_device cuda:0 --headless"


cp "${PY_FILE}" "${BACKUP}"


exp_id=0
for PARAMS in "${PARAM_SETS[@]}"; do
  ((exp_id++))

  echo "==========  Experiment ${exp_id} : ${PARAMS} =========="
  # LOG_DIR="../../logs/rough_lite3/auto_exp_${exp_id}_$(date +%Y%m%d_%H%M%S)"
  # mkdir -p "${LOG_DIR}"

  # 逐个参数写入 Lite3_config.py ——sed
  # shellcheck disable=SC2207
  for KV in ${PARAMS}; do
    key=$(echo "${KV}" | cut -d= -f1)
    val=$(echo "${KV}" | cut -d= -f2)

    # 配置文件里每个参数独占一行，feet_air_time = 2.0，否则 sed 匹配不到。
  echo "sed -i \"s|^\([[:space:]]*${key}[[:space:]]*=[[:space:]]*\).*|\1${val} |\" ${PY_FILE}"

  sed -i "s|^\([[:space:]]*${key}[[:space:]]*=[[:space:]]*\).*|\1${val} |" "${PY_FILE}"
  done

  # 启动训练并把 log 重定向到文件
  # echo "[INFO] launching training to ${LOG_DIR}/train.log"
  ${RUN_CMD} 
    # | tee "${LOG_DIR}/train.log"

  echo "[INFO] Experiment ${exp_id} finished"
done

mv "${BACKUP}" "${PY_FILE}"
echo "[INFO] All experiments done. Config restored."
