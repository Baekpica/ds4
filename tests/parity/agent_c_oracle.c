/* Narrow ds4-agent policy oracle. Function sections let this include the
 * production C source while linking only the prompt and detector leaves used
 * below; no engine is opened and no GPU object is linked. */

#define main ds4_agent_program_main
#include "../../ds4_agent.c"
#undef main

static void print_hex(const char *text, size_t len)
{
    for (size_t i = 0; i < len; i++)
        printf("%02x", (unsigned char)text[i]);
    putchar('\n');
}

static void oracle_usage(void)
{
    fprintf(stderr,
            "usage: agent_c_oracle prompt | datetime WHEN | dsml TEXT | "
            "project THINK CHUNK...\n");
    exit(2);
}

int main(int argc, char **argv)
{
    if (argc == 2 && strcmp(argv[1], "prompt") == 0) {
        char *prompt = agent_build_tools_prompt();
        print_hex(prompt, strlen(prompt));
        free(prompt);
        return 0;
    }

    if (argc == 3 && strcmp(argv[1], "datetime") == 0) {
        char msg[256];
        snprintf(msg, sizeof(msg),
                 "Current local date and time at session start: %s. "
                 "Use this only when date or time matters.", argv[2]);
        print_hex(msg, strlen(msg));
        return 0;
    }

    if (argc == 3 && strcmp(argv[1], "dsml") == 0) {
        bool complete = false;
        bool match = agent_stream_dsml_start_match(argv[2], strlen(argv[2]),
                                                   &complete);
        printf("match=%d complete=%d\n", match ? 1 : 0, complete ? 1 : 0);
        return 0;
    }

    if (argc >= 4 && strcmp(argv[1], "project") == 0) {
        bool thinking = atoi(argv[2]) != 0;
        agent_worker worker = {0};
        if (pthread_mutex_init(&worker.mu, NULL) != 0 || pipe(worker.wake_fd) != 0) {
            perror("agent_c_oracle: projector init");
            return 1;
        }
        agent_token_renderer renderer = {
            .worker = &worker,
            .format_thinking = thinking,
            .format_markdown = false,
            .in_think = thinking,
            .use_color = false,
            .last_output_newline = true,
        };
        agent_dsml_parser parser = {.state = AGENT_DSML_SEARCH};
        agent_stream_renderer stream = {
            .renderer = &renderer,
            .parser = &parser,
            .in_think = thinking,
        };
        for (int i = 3; i < argc; i++)
            agent_stream_text(&stream, argv[i], strlen(argv[i]), false);
        agent_stream_text(&stream, NULL, 0, true);
        renderer_finish(&renderer);
        print_hex(worker.out ? worker.out : "", worker.out_len);
        agent_dsml_parser_free(&parser);
        free(worker.out);
        close(worker.wake_fd[0]);
        close(worker.wake_fd[1]);
        pthread_mutex_destroy(&worker.mu);
        return 0;
    }

    oracle_usage();
    return 2;
}
