#include "../ds4.h"

#include <stdio.h>

static int failures;

#define CHECK(cond, message) do {                                             \
    if (!(cond)) {                                                            \
        fprintf(stderr, "FAIL: %s\n", message);                             \
        failures++;                                                           \
    }                                                                         \
} while (0)

int main(void) {
    /* MTP consumes (h[p], x[p+1]) at decoder position p+1. */
    CHECK(ds4_test_exaone_mtp_position_after_hidden(0) == 1,
          "MTP position must be shifted after its target hidden row");
    CHECK(ds4_test_exaone_mtp_position_after_hidden(262142) == 262143,
          "MTP position shift must remain exact at the 256K boundary");

    /* A cold first cycle is not enough evidence to disable speculation. */
    CHECK(!ds4_test_exaone_mtp_should_quench(11, 0, 80.0, 2000.0, 500.0),
          "warm-up must not quench before twelve verifier cycles");

    /* 12 cycles + 9 accepted drafts commit 21 tokens. At an 80 ms plain
     * decode baseline, 1500 ms of verify/draft work still saves time. */
    CHECK(!ds4_test_exaone_mtp_should_quench(12, 9, 80.0, 1450.0, 50.0),
          "profitable speculation must remain enabled");

    /* The same acceptance with 1900 ms of work is slower than 21 ordinary
     * target decodes, including a small noise margin. */
    CHECK(ds4_test_exaone_mtp_should_quench(12, 9, 80.0, 1850.0, 50.0),
          "cumulative negative speculation must quench");

    CHECK(!ds4_test_exaone_mtp_should_quench(100, 0, 0.0, 99999.0, 99999.0),
          "missing baseline must not make a policy decision");

    if (failures) return 1;
    puts("exaone MTP auto-quench policy: all checks passed");
    return 0;
}
