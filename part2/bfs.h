#ifndef __BFS_H__
#define __BFS_H__

#define USE_HYBRID_FUNCTION
// #define DEBUG_HYBRID
// #define DEBUG

struct graph;

struct solution
{
    int* distances;
};

struct vertex_set {
    int  count;
    int  alloc_count;
    int* present;
};

struct graph_info {
    size_t frontier_edges;
    size_t frontier_vertices;
    size_t unexplored_edges;
};


void bfs_bottom_up(graph* graph, solution* sol);
void bfs_top_down(graph* graph, solution* sol);
void bfs_hybrid(graph* graph, solution* sol);


void reference_bfs_bottom_up(graph* graph, solution* sol);
void reference_bfs_top_down(graph* graph, solution* sol);
void reference_bfs_hybrid(graph* graph, solution* sol);


#endif
