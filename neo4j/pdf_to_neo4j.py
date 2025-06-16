#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PDF文档处理并实时导入Neo4j数据库的脚本
功能：
1. 实时提取PDF文档中的文本内容并写入Neo4j
2. 对文本进行分块和预处理
3. 提取实体和关系
4. 实时将数据写入Neo4j图数据库
"""

import os
import PyPDF2
import re
from typing import List, Dict, Tuple, Generator
from neo4j import GraphDatabase
from tqdm import tqdm
import logging
from datetime import datetime
import jieba
import jieba.posseg as pseg

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'pdf_processing_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

class PDFProcessor:
    """PDF文档处理器类"""
    
    def __init__(self, pdf_path: str, neo4j_importer):
        """
        初始化PDF处理器
        
        Args:
            pdf_path: PDF文件路径
            neo4j_importer: Neo4j导入器实例
        """
        self.pdf_path = pdf_path
        self.neo4j_importer = neo4j_importer
        self.current_page = 0
        self.total_pages = 0
        
    def process_pdf(self, doc_id: str, title: str):
        """
        处理PDF文件并实时写入Neo4j
        
        Args:
            doc_id: 文档ID
            title: 文档标题
        """
        try:
            with open(self.pdf_path, 'rb') as file:
                # 创建PDF阅读器对象
                pdf_reader = PyPDF2.PdfReader(file)
                self.total_pages = len(pdf_reader.pages)
                
                # 创建文档节点
                self.neo4j_importer.create_document(doc_id, title)
                
                # 遍历所有页面并实时处理
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    self.current_page = page_num
                    text = page.extract_text()
                    
                    # 处理页面文本
                    self._process_page(doc_id, page_num, text)
                    
                    # 每处理完一页就提交事务
                    self.neo4j_importer.commit_transaction()
                    
                logging.info(f"成功处理完 {self.pdf_path} 的所有页面")
                
        except Exception as e:
            logging.error(f"PDF处理失败: {str(e)}")
            raise
            
    def _process_page(self, doc_id: str, page_num: int, text: str):
        """
        处理单个页面的文本
        
        Args:
            doc_id: 文档ID
            page_num: 页码
            text: 页面文本
        """
        # 创建页面节点
        page_id = f"{doc_id}_page_{page_num}"
        self.neo4j_importer.create_page(doc_id, page_id, page_num)
        
        # 分割文本为段落
        paragraphs = re.split(r'\n\s*\n', text)
        
        for para_idx, para in enumerate(paragraphs, 1):
            if not para.strip():
                continue
                
            # 创建段落节点
            para_id = f"{page_id}_para_{para_idx}"
            self.neo4j_importer.create_paragraph(page_id, para_id, para_idx, para)
            
            # 提取实体和关系
            self._extract_entities_and_relations(para_id, para)

    def _extract_entities_and_relations(self, para_id: str, text: str):
        """
        从文本中提取实体和关系
        
        Args:
            para_id: 段落ID
            text: 段落文本
        """
        # 使用jieba进行词性标注
        words = pseg.cut(text)
        
        # 提取实体和关系
        entities = []
        current_entity = []
        current_entity_type = None
        
        for word, flag in words:
            # 根据词性标注提取实体
            if flag.startswith('n'):  # 名词
                if not current_entity:
                    current_entity_type = 'Concept'
                current_entity.append(word)
            elif flag.startswith('v'):  # 动词
                if current_entity:
                    # 保存当前实体
                    if current_entity:
                        entity_text = ''.join(current_entity)
                        entities.append((entity_text, current_entity_type))
                        # 创建实体节点
                        self.neo4j_importer.create_entity(para_id, entity_text, current_entity_type)
                    current_entity = []
                    current_entity_type = 'Action'
                current_entity.append(word)
            else:
                if current_entity:
                    entity_text = ''.join(current_entity)
                    entities.append((entity_text, current_entity_type))
                    # 创建实体节点
                    self.neo4j_importer.create_entity(para_id, entity_text, current_entity_type)
                    current_entity = []
                    current_entity_type = None

class Neo4jImporter:
    """Neo4j数据库导入器类"""
    
    def __init__(self, uri: str, user: str, password: str):
        """
        初始化Neo4j导入器
        
        Args:
            uri: Neo4j数据库URI
            user: 用户名
            password: 密码
        """
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.session = self.driver.session()
        self.transaction = None
        
    def close(self):
        """关闭数据库连接"""
        if self.transaction:
            self.transaction.close()
        if self.session:
            self.session.close()
        self.driver.close()
        
    def start_transaction(self):
        """开始新的事务"""
        if self.transaction:
            self.transaction.close()
        self.transaction = self.session.begin_transaction()
        
    def commit_transaction(self):
        """提交当前事务并开始新的事务"""
        if self.transaction:
            self.transaction.commit()
        self.start_transaction()
        
    def create_constraints(self):
        """创建必要的数据库约束"""
        with self.driver.session() as session:
            # 创建各种节点的唯一性约束
            constraints = [
                ("Document", "id"),
                ("Page", "id"),
                ("Paragraph", "id"),
                ("Entity", "id")
            ]
            
            for label, property in constraints:
                session.run(f"""
                    CREATE CONSTRAINT {label.lower()}_{property} IF NOT EXISTS
                    FOR (n:{label}) REQUIRE n.{property} IS UNIQUE
                """)
            
            logging.info("数据库约束创建完成")
    
    def create_document(self, doc_id: str, title: str):
        """创建文档节点"""
        self.transaction.run("""
            MERGE (d:Document {id: $doc_id})
            SET d.title = $title,
                d.created_at = datetime()
        """, doc_id=doc_id, title=title)
        
    def create_page(self, doc_id: str, page_id: str, page_num: int):
        """创建页面节点"""
        self.transaction.run("""
            MATCH (d:Document {id: $doc_id})
            MERGE (p:Page {id: $page_id})
            SET p.page_number = $page_num
            MERGE (d)-[:CONTAINS_PAGE]->(p)
        """, doc_id=doc_id, page_id=page_id, page_num=page_num)
        
    def create_paragraph(self, page_id: str, para_id: str, para_num: int, content: str):
        """创建段落节点"""
        self.transaction.run("""
            MATCH (p:Page {id: $page_id})
            MERGE (para:Paragraph {id: $para_id})
            SET para.paragraph_number = $para_num,
                para.content = $content
            MERGE (p)-[:CONTAINS_PARAGRAPH]->(para)
        """, page_id=page_id, para_id=para_id, 
            para_num=para_num, content=content)
            
    def create_entity(self, para_id: str, entity_text: str, entity_type: str):
        """创建实体节点"""
        self.transaction.run("""
            MATCH (para:Paragraph {id: $para_id})
            MERGE (e:Entity {id: $entity_id})
            SET e.text = $entity_text,
                e.type = $entity_type
            MERGE (para)-[:CONTAINS_ENTITY]->(e)
        """, para_id=para_id, 
            entity_id=f"{para_id}_entity_{entity_text}",
            entity_text=entity_text,
            entity_type=entity_type)

def main():
    """主函数"""
    # Neo4j连接配置
    NEO4J_URI = "bolt://localhost:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "your_password"  # 请修改为实际的密码
    
    # PDF文件路径
    PDF_PATH = "../实用供热空调设计手册第二版(上册).pdf"
    
    try:
        # 初始化Neo4j导入器
        neo4j_importer = Neo4jImporter(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
        
        # 创建数据库约束
        neo4j_importer.create_constraints()
        
        # 开始事务
        neo4j_importer.start_transaction()
        
        # 初始化PDF处理器并处理文档
        pdf_processor = PDFProcessor(PDF_PATH, neo4j_importer)
        doc_id = "heating_manual_vol1"
        title = "实用供热空调设计手册第二版(上册)"
        pdf_processor.process_pdf(doc_id, title)
        
        # 提交最后的事务
        neo4j_importer.commit_transaction()
        
        logging.info("文档处理和数据导入完成")
        
    except Exception as e:
        logging.error(f"处理过程中发生错误: {str(e)}")
        if 'neo4j_importer' in locals() and neo4j_importer.transaction:
            neo4j_importer.transaction.rollback()
    finally:
        if 'neo4j_importer' in locals():
            neo4j_importer.close()

if __name__ == "__main__":
    main() 