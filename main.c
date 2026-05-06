#include <stdio.h>

typedef struct {
    signed int b : 4;
} bias;

bias init_bias(bias b) {
    b.b = 0;
    return b;
}

typedef struct {
    bias tb[16];
} b_64;

b_64 init_b64(b_64 tb) {
    for (int i = 0; i < sizeof(tb.tb)/sizeof(tb.tb[0]); ++i) {
        tb.tb[i] = init_bias(tb.tb[i]);
    }
    return tb;
}

int main() {

    /*
        Use Two's Complement:
        - Reverse the binary of the positive number and add 1 to get the negative of the same number.
        . Tiles across CPU/GPU nicely:
        . 64bit is most common, so we create a 64bit-native system.
        
        0000                : 4bit
        0000 0000           : 8bit
        0000 0000 0000 0000 : 16bit

        Use an expanded 4bit version to tile across 64bit:
            64 / 4 = 16
        
        There are 2 modes of operation:
        - pass through normally
        - inhibitor
        These modes of operation are demonstrated by the being either positive or negative values

        4bit biases only. Cyclical biases mean that a 4bit structure can be maintained.
        1111 + 0001 = -1 + 1 = 0

        For real intelligence (core):
        error_t = x_t - x_hat_t
        x_hat_(t+1) = f(state_t, error_t)

        predict -> compare -> update
        . compression (build an internal model that is shorter than its experience)
        . hierarchy (each level predicts the level below - input prediction)
        . temporal credit assignment (knowing which past action caused a current outcome)
    */

    return 0;
}