use lambdas::ZNode;



type ZipTrieIdx = usize;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct ZipTrieNode {
    present: bool,
    func: Option<ZipTrieIdx>,
    arg: Option<ZipTrieIdx>,
    body: Option<ZipTrieIdx>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ZipTree {
    root: ZipTrieIdx,
    trie: Vec<ZipTrieNode>,
}

#[derive(Debug, Clone)]
pub struct ZipTrieSlice<'a> {
    trie: &'a ZipTree,
    start: ZipTrieIdx,
}

impl ZipTree {
    pub fn new(mut zippers: Vec<Vec<ZNode>>) -> Self {
        let mut trie = vec![];
        zippers.sort();
        let index = add_to_trie(&mut trie, &zippers[..], 0);
        ZipTree {root: index.unwrap(), trie}
    }
}

impl <'a> ZipTrieSlice<'a> {
    pub fn new(trie: &'a ZipTree) -> Self {
        ZipTrieSlice {trie, start: trie.root}
    }

    pub fn is_present(&self) -> bool {
        self.trie.trie[self.start].present
    }

    pub fn func(&self) -> Option<ZipTrieSlice<'a>> {
        let idx = self.trie.trie[self.start].func?;
        Some(ZipTrieSlice {trie: self.trie, start: idx})
    }

    pub fn arg(&self) -> Option<ZipTrieSlice<'a>> {
        let idx = self.trie.trie[self.start].arg?;
        Some(ZipTrieSlice {trie: self.trie, start: idx})
    }

    pub fn body(&self) -> Option<ZipTrieSlice<'a>> {
        let idx = self.trie.trie[self.start].body?;
        Some(ZipTrieSlice {trie: self.trie, start: idx})
    }
}

fn add_to_trie(
    trie: &mut Vec<ZipTrieNode>,
    zippers: &[Vec<ZNode>],
    depth: usize,
) -> Option<ZipTrieIdx> {
    if zippers.is_empty() {
        return None;
    }
    let mut zippers = zippers;
    let present = zippers[0].len() == depth;
    if present {
        zippers = &zippers[1..];
    }
    let mut ztnode = ZipTrieNode {
        present,
        func: None,
        arg: None,
        body: None,
    };
    let mut fire = |
        current: ZNode,
        start_idx: usize,
        end_idx: usize,
    | {
        let loc_zippers = &zippers[start_idx..end_idx];
        let idx = add_to_trie(trie, loc_zippers, depth + 1);
        match current {
            ZNode::Func => {
                ztnode.func = idx;
            }
            ZNode::Arg => {
                ztnode.arg = idx;
            }
            ZNode::Body => {
                ztnode.body = idx;
            }
        }
    };

    let mut start_idx = None;
    let mut current = None;
    for (idx, zip) in zippers.iter().enumerate() {
        let znode = zip[depth].clone();
        if current == Some(znode.clone()) {
            // we are at the same node, so we can just continue
            continue;
        }
        if current.is_some() {
            fire(current.clone().unwrap(), start_idx.unwrap(), idx);
        }
        current = Some(znode);
        start_idx = Some(idx);
    }
    if let Some(current) = current {
        fire(current, start_idx.unwrap(), zippers.len());
    }
    
    let idx = trie.len();
    trie.push(ztnode);
    Some(idx)
}
