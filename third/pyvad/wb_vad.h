#ifndef wb_vad_h 
#define wb_vad_h 
 
/****************************************************************************** 
*                         INCLUDE FILES 
******************************************************************************/ 
#include "stdint.h"
#include "wb_vad_c.h" 
 
/****************************************************************************** 
*                         DEFINITION OF DATA TYPES 
******************************************************************************/ 
 
typedef struct 
{ 
    float bckr_est[COMPLEN];    /* background noise estimate */ 
    float ave_level[COMPLEN];   /* averaged input components for stationary */ 
    /* estimation */ 
    float old_level[COMPLEN];   /* input levels of the previous frame */ 
    float sub_level[COMPLEN];   /* input levels calculated at the end of a frame (lookahead) */ 
    float a_data5[F_5TH_CNT][2];        /* memory for the filter bank */ 
    float a_data3[F_3TH_CNT];   /* memory for the filter bank */ 
 
    int16_t burst_count;         /* counts length of a speech burst */ 
    int16_t hang_count;          /* hangover counter */ 
    int16_t stat_count;          /* stationary counter */ 
 
    /* Note that each of the following two variables holds 15 flags. Each flag reserves 1 bit of the variable. The newest flag is 
     * in the bit 15 (assuming that LSB is bit 1 and MSB is bit 16). */ 
    int16_t vadreg;              /* flags for intermediate VAD decisions */ 
    int16_t pitch_tone;          /* flags for pitch and tone detection */ 
 
    int16_t sp_est_cnt;          /* counter for speech level estimation */ 
    float sp_max;               /* maximum level */ 
    int16_t sp_max_cnt;          /* counts frames that contains speech */ 
    float speech_level;         /* estimated speech level */ 
    double prev_pow_sum;        /* power of previous frame */ 
 
    float level[COMPLEN]; 
    float prevLevel[COMPLEN]; 

    float pow_low;
    float pow_pitch_tone_thr;
} VadVars; 
 
/* 
******************************************************************************** 
*                         DECLARATION OF PROTOTYPES 
******************************************************************************** 
*/ 
int wb_vad_init(VadVars ** st); 
int wb_vad_reset(VadVars * st); 
void wb_vad_exit(VadVars ** st); 
void wb_vad(VadVars * st, const int16_t in_buf[], int* res, double* power_sum); 
void wb_vad_pitch_tone_detection(VadVars * st, float p_gain); 
void wb_vad_set_pow_low(VadVars * st, float pl);
void wb_vad_set_pow_pitch_tone_thr(VadVars * st, float pptt);
 
#endif 
