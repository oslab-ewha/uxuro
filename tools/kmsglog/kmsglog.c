#include <stdlib.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>
#include <errno.h>

#define BUF_SIZE	16384

static int	finished = 0;

static void
sighandler(int signum, siginfo_t *info, void *ptr)
{
	finished = 1;
}

static void
setup_signals(void)
{
	struct sigaction	sig = { 0 };

	sig.sa_sigaction = sighandler;
	sig.sa_flags = SA_SIGINFO;
	sigaction(SIGTERM, &sig, NULL);
	sigaction(SIGINT, &sig, NULL);
}

static char *
find_nr(char *buf, unsigned nbuf)
{
	unsigned	i;

	for (i = 0; i < nbuf; i++) {
		if (buf[i] == '\n')
			return buf + i;
	}
	return NULL;
}

static void
process_kmsg(char *buf)
{
	char	*semi;

	semi = strchr(buf, ';');
	if (semi == NULL)
		return;
	if (strncmp(semi + 1, "uXuA:", 5) != 0)
		return;
	printf("%s\n", semi + 6);
}

static void
dump_kmsg(int fd)
{
	char	buf[BUF_SIZE];
	unsigned	nbufs = 0;

	while (!finished) {
		ssize_t	nread;
		char	*nr;

		nread = read(fd, buf + nbufs, BUF_SIZE - nbufs);
		if (nread < 0) {
			if (errno != EAGAIN) {
				fprintf(stderr, "failed to read: err: %d\n", errno);
				break;
			}
			usleep(10);
			continue;
		}

		while (nread > 0) {
			char	*found;

			found = find_nr(buf + nbufs, nread);
			if (found) {
				unsigned	remain;

				*found = '\0';
				process_kmsg(buf);
				remain = nbufs + nread - (found - buf) - 1;
				if (remain > 0)
					memcpy(buf, found + 1, remain);
				nbufs = 0;
				nread = remain;
			}
			else {
				nbufs += nread;
				break;
			}
		}
	}
}

int
main(int argc, char *argv[])
{
	ssize_t	ret;
	int	fd;

	fd = open("/dev/kmsg", O_RDONLY | O_NONBLOCK);
	if (fd < 0) {
		fprintf(stderr, "failed to open /dev/kmsg\n");
		exit(1);
	}
	lseek(fd, 0, SEEK_END);

	setup_signals();

	dump_kmsg(fd);

	close(fd);
	return 0;
}
