#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstddef>
#include <omp.h>

#include <algorithm>
#include <functional>
#include <numeric>
#include <vector>

#include "CycleTimer.h"
#include "bfs.h"
#include "graph.h"

#define ROOT_NODE_ID 0
#define NOT_VISITED_MARKER -1

#include <limits.h>     /* for CHAR_BIT */

#define BITMASK(b) (1 << ((b) % CHAR_BIT))
#define BITSLOT(b) ((b) / CHAR_BIT)
#define BITSET(a, b) ((a)[BITSLOT(b)] |= BITMASK(b))
#define BITCLEAR(a, b) ((a)[BITSLOT(b)] &= ~BITMASK(b))
#define BITTEST(a, b) ((a)[BITSLOT(b)] & BITMASK(b))
#define BITNSLOTS(nb) ((nb + CHAR_BIT - 1) / CHAR_BIT)

void vertex_set_clear(vertex_set* list) {
    list->count = 0;
}

void vertex_set_init(vertex_set* list, int count) {
    list->alloc_count = count;
    list->present = (int*)malloc(sizeof(int) * list->alloc_count);
    vertex_set_clear(list);
}

#define LOOPS_PER_THREAD 64

// We note that in step s of the BFS, everything that should be added to the new
// frontier has a neighbor with distance s (and everything in the frontier has
// distance s), and no neighbors with distance less than s

void bottom_up_step(
    graph* g,
    char* frontier,
    char* new_frontier,
    int* distances,
    int step,
    graph_info* info)
{
    int local_frontier_size = 0, local_explored_edges = 0;

    #pragma omp parallel for reduction(+:local_frontier_size, local_explored_edges)
    for (int i = 0; i < g->num_nodes / LOOPS_PER_THREAD + 1; i++) {
        for (int j = 0; j < LOOPS_PER_THREAD; j++) {
            int node = i * LOOPS_PER_THREAD + j;
            if (node < g->num_nodes && distances[node] == NOT_VISITED_MARKER) {
                int start_edge = g->incoming_starts[node];
                int end_edge = (node == g->num_nodes - 1) ? g->num_edges : g->incoming_starts[node + 1];

                for (int neighbor = start_edge; neighbor < end_edge; neighbor++) {
                    int incoming = g->incoming_edges[neighbor];

                    if (BITTEST(frontier, incoming)) {
                        distances[node] = distances[incoming] + 1;
                        BITSET(new_frontier, node);
                        local_frontier_size++;

                        break;
                    }
                }
                local_explored_edges += (end_edge - start_edge);
            }
        }
    }

    info->frontier_vertices = local_frontier_size;
    info->unexplored_edges -= local_explored_edges;
    info->frontier_edges = local_explored_edges;
}

void bfs_bottom_up(graph* graph, solution* sol)
{
    // 15-418/618 students:
    //
    // You will need to implement the "bottom up" BFS here as
    // described in the handout.
    //
    // As a result of your code's execution, sol.distances should be
    // correctly populated for all nodes in the graph.
    //
    // As was done in the top-down case, you may wish to organize your
    // code by creating subroutine bottom_up_step() that is called in
    // each step of the BFS process.

    char list1[BITNSLOTS(graph->num_nodes)], list2[BITNSLOTS(graph->num_nodes)];

    char* frontier = list1, *new_frontier = list2;

    memset(frontier, 0, BITNSLOTS(graph->num_nodes));
    memset(new_frontier, 0, BITNSLOTS(graph->num_nodes));

    // initialize all nodes to NOT_VISITED
    for (int i=0; i<graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    // setup frontier with the root node
    BITSET(frontier, ROOT_NODE_ID);
    sol->distances[ROOT_NODE_ID] = 0;

    int step = 0;
    graph_info info;

    info.unexplored_edges = graph->num_edges;
    info.frontier_vertices = 1;
    info.frontier_edges = 0;

    while (info.frontier_vertices != 0) {

#ifdef DEBUG
        double start_time = CycleTimer::currentSeconds();
#endif

        memset(new_frontier, 0, BITNSLOTS(graph->num_nodes));

        bottom_up_step(graph, frontier, new_frontier, sol->distances, step++, &info);

#ifdef DEBUG
        double end_time = CycleTimer::currentSeconds();
        printf("frontier=%-10d %.4f sec\n", 1, end_time - start_time);
#endif

        // swap pointers
        // frontier.swap(new_frontier);
        char* tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;
    }
}


// Take one step of "top-down" BFS.  For each vertex on the frontier,
// follow all outgoing edges, and add all neighboring vertices to the
// new_frontier.
void top_down_step(
    graph* g,
    char* frontier,
    char* new_frontier,
    int* distances,
    graph_info* info)
{
    int local_frontier_size = 0, local_frontier_edges = 0, local_explored_edges = 0;

    #pragma omp parallel for reduction(+: local_explored_edges, local_frontier_size, local_frontier_edges)
    for (int i = 0; i < g->num_nodes / LOOPS_PER_THREAD + 1; i++) {
        for (int j = 0; j < LOOPS_PER_THREAD; j++) {
            int node = i * LOOPS_PER_THREAD + j;
            if (node < g->num_nodes && frontier[node]) {
                int start_edge = g->outgoing_starts[node];
                int end_edge = (node == g->num_nodes - 1) ? g->num_edges : g->outgoing_starts[node + 1];

                // attempt to add all neighbors to the new frontier
                for (int neighbor=start_edge; neighbor<end_edge; neighbor++) {
                    int outgoing = g->outgoing_edges[neighbor];

                    if (distances[outgoing] == NOT_VISITED_MARKER) {
                        distances[outgoing] = distances[node] + 1;
                        new_frontier[outgoing] = 1;
                        local_frontier_size++;
                        local_frontier_edges +=
                                ((outgoing == g->num_nodes - 1) ? g->num_edges : g->outgoing_starts[outgoing + 1])
                                - g->outgoing_starts[outgoing];
                    }
                }
                local_explored_edges += (end_edge - start_edge);
            }
        }
    }

    info->frontier_vertices = local_frontier_size;
    info->frontier_edges = local_frontier_edges;
    info->unexplored_edges -= local_explored_edges;

}

// Implements top-down BFS.
//
// Result of execution is that, for each node in the graph, the
// distance to the root is stored in sol.distances.
void bfs_top_down(graph* graph, solution* sol) {

    char list1[graph->num_nodes], list2[graph->num_nodes];

    char* frontier = list1, *new_frontier = list2;

    memset(frontier, 0, graph->num_nodes);
    memset(new_frontier, 0, graph->num_nodes);


    // initialize all nodes to NOT_VISITED
    for (int i=0; i<graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    // setup frontier with the root node
    frontier[ROOT_NODE_ID] = 1;
    sol->distances[ROOT_NODE_ID] = 0;

    graph_info info;

    info.unexplored_edges = graph->num_edges;
    info.frontier_vertices = 10;
    info.frontier_edges = 0;

    while (info.frontier_vertices != 0) {

#ifdef DEBUG
        double start_time = CycleTimer::currentSeconds();
#endif

        memset(new_frontier, 0, graph->num_nodes);

        top_down_step(graph, frontier, new_frontier, sol->distances, &info);


#ifdef DEBUG
        double end_time = CycleTimer::currentSeconds();
        printf("frontier=%-10d %.4f sec\n", info.frontier_vertices, end_time - start_time);
#endif

        // swap pointers
        char* tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;
    }
}

#define ALPHA 14
#define BETA 24

void bfs_hybrid(graph* graph, solution* sol)
{
    // 15-418/618 students:
    //
    // You will need to implement the "bottom up" BFS here as
    // described in the handout.
    //
    // As a result of your code's execution, sol.distances should be
    // correctly populated for all nodes in the graph.
    //
    // As was done in the top-down case, you may wish to organize your
    // code by creating subroutine bottom_up_step() that is called in
    // each step of the BFS process.

    char list1[graph->num_nodes], list2[graph->num_nodes], bits1[BITNSLOTS(graph->num_nodes)], bits2[BITNSLOTS(graph->num_nodes)];

    char *frontier = list1, *new_frontier = list2, *bits_frontier = bits1, *new_bits_frontier = bits2;

    memset(frontier, 0, graph->num_nodes);
    memset(new_frontier, 0, graph->num_nodes);
    memset(bits_frontier, 0, BITNSLOTS(graph->num_nodes));
    memset(new_bits_frontier, 0, BITNSLOTS(graph->num_nodes));

    // initialize all nodes to NOT_VISITED
    for (int i=0; i<graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    // setup frontier with the root node
    // frontier->present[frontier->count++] = ROOT_NODE_ID;
    frontier[ROOT_NODE_ID] = 1;
    sol->distances[ROOT_NODE_ID] = 0;

    int step = 0;
    graph_info info;

    char use_top_down = true;

    info.unexplored_edges = graph->num_edges;
    info.frontier_vertices = 1;
    info.frontier_edges = 0;

    while (info.frontier_vertices != 0) {

#ifdef DEBUG_HYBRID
        double start_time = CycleTimer::currentSeconds();
#endif

        if (use_top_down && info.frontier_edges > info.unexplored_edges / ALPHA) {
            use_top_down = false;
            for (int i = 0; i < graph->num_nodes; i++) {
                if (frontier[i]) {
                    BITSET(bits_frontier, i);
                }
            }
        }
        else if (!use_top_down && info.frontier_vertices < graph->num_nodes / BETA) {
            use_top_down = true;
            for (int i = 0; i < graph->num_nodes; i++) {
                frontier[i] = BITTEST(bits_frontier, i) ? 1 : 0;
            }
        }

        if (use_top_down) {
            memset(new_frontier, 0, graph->num_nodes);
            top_down_step(graph, frontier, new_frontier, sol->distances, &info);

            // swap pointers
            char* tmp = frontier;
            frontier = new_frontier;
            new_frontier = tmp;
        }
        else {
            memset(new_bits_frontier, 0, BITNSLOTS(graph->num_nodes));
            bottom_up_step(graph, bits_frontier, new_bits_frontier, sol->distances, step, &info);

            // swap pointers
            char* tmp = bits_frontier;
            bits_frontier = new_bits_frontier;
            new_bits_frontier = tmp;
        }

        step++;


#ifdef DEBUG_HYBRID
        double end_time = CycleTimer::currentSeconds();
        printf("frontier=%-10d %.4f sec\n", info.frontier_vertices, end_time - start_time);
#endif


    }
}

